/**
 * GitHub REST API wrapper for reading/writing JSON files in the repo.
 * All persistent data lives in the /data directory of the repo.
 */
class GitHubAPI {
  constructor() {
    this.owner = null;
    this.repo = null;
    this.token = null;
    this.baseUrl = 'https://api.github.com';
    this._shaCache = {}; // path -> sha
  }

  /** Configure the API with owner, repo, and PAT */
  configure(owner, repo, token) {
    this.owner = owner;
    this.repo = repo;
    this.token = token;
  }

  /** Check if the API is configured */
  isConfigured() {
    return !!(this.owner && this.repo && this.token);
  }

  /** Build the API URL for a file path */
  _url(path) {
    return `${this.baseUrl}/repos/${this.owner}/${this.repo}/contents/${path}`;
  }

  /** Common headers for API requests */
  _headers() {
    return {
      'Authorization': `token ${this.token}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json'
    };
  }

  /**
   * Read a file from the repo. Returns parsed JSON content.
   * Returns null if the file doesn't exist (404).
   * Caches the SHA for subsequent writes.
   */
  async readFile(path) {
    if (!this.isConfigured()) throw new Error('GitHub API not configured');

    const response = await fetch(this._url(path), {
      headers: this._headers()
    });

    if (response.status === 404) {
      return null;
    }

    if (response.status === 403) {
      const remaining = response.headers.get('X-RateLimit-Remaining');
      if (remaining === '0') {
        const resetTime = new Date(response.headers.get('X-RateLimit-Reset') * 1000);
        throw new Error(`GitHub API rate limit exceeded. Resets at ${resetTime.toLocaleTimeString()}`);
      }
      throw new Error('GitHub API access forbidden. Check your token permissions.');
    }

    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    this._shaCache[path] = data.sha;

    // Decode base64 content
    const content = atob(data.content.replace(/\n/g, ''));
    // Handle UTF-8 properly
    const bytes = new Uint8Array(content.length);
    for (let i = 0; i < content.length; i++) {
      bytes[i] = content.charCodeAt(i);
    }
    const decoded = new TextDecoder('utf-8').decode(bytes);

    try {
      return JSON.parse(decoded);
    } catch {
      return decoded; // Return raw string if not JSON
    }
  }

  /**
   * Write (create or update) a file in the repo.
   * Automatically fetches the latest SHA before updating to avoid conflicts.
   */
  async writeFile(path, content, message) {
    if (!this.isConfigured()) throw new Error('GitHub API not configured');

    // Always fetch the latest SHA before writing to avoid conflicts
    let sha = null;
    try {
      const existingResponse = await fetch(this._url(path), {
        headers: this._headers()
      });
      if (existingResponse.ok) {
        const existingData = await existingResponse.json();
        sha = existingData.sha;
      }
    } catch {
      // File doesn't exist yet, that's fine
    }

    const jsonStr = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
    // Encode to UTF-8 then base64
    const encoder = new TextEncoder();
    const utf8Bytes = encoder.encode(jsonStr);
    let binary = '';
    for (let i = 0; i < utf8Bytes.length; i++) {
      binary += String.fromCharCode(utf8Bytes[i]);
    }
    const encoded = btoa(binary);

    const body = {
      message: message || `Update ${path}`,
      content: encoded
    };

    if (sha) {
      body.sha = sha;
    }

    const response = await fetch(this._url(path), {
      method: 'PUT',
      headers: this._headers(),
      body: JSON.stringify(body)
    });

    if (response.status === 409) {
      throw new Error('Write conflict — another change was made. Please try again.');
    }

    if (response.status === 403) {
      const remaining = response.headers.get('X-RateLimit-Remaining');
      if (remaining === '0') {
        const resetTime = new Date(response.headers.get('X-RateLimit-Reset') * 1000);
        throw new Error(`Rate limit exceeded. Resets at ${resetTime.toLocaleTimeString()}`);
      }
      throw new Error('GitHub API access forbidden. Check your token permissions.');
    }

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(`GitHub API write error: ${response.status} — ${errBody.message || response.statusText}`);
    }

    const result = await response.json();
    this._shaCache[path] = result.content.sha;
    return result;
  }

  /**
   * List files in a directory. Returns array of {name, path, sha}.
   * Returns empty array if directory doesn't exist.
   */
  async listFiles(path) {
    if (!this.isConfigured()) throw new Error('GitHub API not configured');

    const response = await fetch(this._url(path), {
      headers: this._headers()
    });

    if (response.status === 404) return [];

    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    if (!Array.isArray(data)) return []; // It's a file, not a directory
    return data.map(f => ({ name: f.name, path: f.path, sha: f.sha }));
  }

  /**
   * Delete a file from the repo.
   */
  async deleteFile(path, message) {
    if (!this.isConfigured()) throw new Error('GitHub API not configured');

    // Get current SHA
    const response = await fetch(this._url(path), {
      headers: this._headers()
    });
    if (!response.ok) return; // Already gone

    const data = await response.json();
    const sha = data.sha;

    await fetch(this._url(path), {
      method: 'DELETE',
      headers: this._headers(),
      body: JSON.stringify({
        message: message || `Delete ${path}`,
        sha
      })
    });
  }

  /**
   * Test the connection by reading the repo info.
   */
  async testConnection() {
    const response = await fetch(`${this.baseUrl}/repos/${this.owner}/${this.repo}`, {
      headers: this._headers()
    });
    if (!response.ok) {
      throw new Error(`Cannot access repo: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  }

  // ----------  GitHub Actions helpers  ----------

  /**
   * Trigger a workflow_dispatch event.
   * @param {string} workflowFile - e.g. "run-model.yml"
   * @param {string} ref - branch name, default "main"
   * @param {Object} inputs - workflow dispatch inputs
   */
  async triggerWorkflowDispatch(workflowFile, ref = 'main', inputs = {}) {
    if (!this.isConfigured()) throw new Error('GitHub API not configured');

    const url = `${this.baseUrl}/repos/${this.owner}/${this.repo}/actions/workflows/${workflowFile}/dispatches`;
    const response = await fetch(url, {
      method: 'POST',
      headers: this._headers(),
      body: JSON.stringify({ ref, inputs })
    });

    if (response.status === 204 || response.status === 200) return true;

    if (response.status === 403) {
      throw new Error(
        'Workflow dispatch failed: 403 — Your PAT is missing the "workflow" scope. ' +
        'Go to GitHub → Settings → Developer settings → Personal access tokens → ' +
        'edit your token and enable the "workflow" scope, then save and re-enter it here.'
      );
    }

    if (response.status === 404) {
      throw new Error('Workflow not found. Make sure the workflow YAML is pushed to the repo and the PAT has the "workflow" scope.');
    }

    const errBody = await response.json().catch(() => ({}));
    throw new Error(`Workflow dispatch failed: ${response.status} — ${errBody.message || response.statusText}`);
  }

  /**
   * Get recent runs for a specific workflow file.
   * @param {string} workflowFile - e.g. "run-model.yml"
   * @param {number} perPage - how many runs to fetch
   */
  async getWorkflowRuns(workflowFile, perPage = 5) {
    if (!this.isConfigured()) throw new Error('GitHub API not configured');

    const url = `${this.baseUrl}/repos/${this.owner}/${this.repo}/actions/workflows/${workflowFile}/runs?per_page=${perPage}`;
    const response = await fetch(url, { headers: this._headers() });

    if (response.status === 404) return { workflow_runs: [] };
    if (!response.ok) {
      throw new Error(`Failed to get workflow runs: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  }

  /**
   * Get a single workflow run by ID.
   */
  async getWorkflowRun(runId) {
    if (!this.isConfigured()) throw new Error('GitHub API not configured');

    const url = `${this.baseUrl}/repos/${this.owner}/${this.repo}/actions/runs/${runId}`;
    const response = await fetch(url, { headers: this._headers() });
    if (!response.ok) {
      throw new Error(`Failed to get run: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  }
}

// Singleton instance
const githubAPI = new GitHubAPI();
