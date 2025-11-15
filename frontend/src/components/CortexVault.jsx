// frontend/src/components/CortexVault.jsx
import React, { useState, useCallback } from 'react';
import { Upload, FileText, Image, Video, Database, FolderTree, CheckCircle, AlertCircle, Loader2, X, LogIn, UserPlus, Eye, EyeOff } from 'lucide-react';

/*
  CortexVault React component (calls FastAPI backend at http://localhost:8000)
  Make sure backend is running.
*/

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

const CortexVault = () => {
  const [authState, setAuthState] = useState('login'); // 'login', 'signup', 'app'
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(false);

  // Auth form states
  const [loginData, setLoginData] = useState({ email: '', password: '' });
  const [signupData, setSignupData] = useState({ name: '', email: '', password: '', confirmPassword: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [authError, setAuthError] = useState('');

  // App states
  const [files, setFiles] = useState([]); // {file, tmp_path, info, status, customFolder}
  const [jsonInput, setJsonInput] = useState('');
  const [jsonAnalysis, setJsonAnalysis] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({ showThumbnails: true, showDocPreview: true });

  // ---------- Auth (local simulated) ----------
  const handleLogin = async (e) => {
    e && e.preventDefault();
    setAuthError('');
    setAuthLoading(true);
    await new Promise(r => setTimeout(r, 800));
    if (loginData.email && loginData.password) {
      setUser({ name: loginData.email.split('@')[0], email: loginData.email });
      setAuthState('app');
    } else {
      setAuthError('Please fill in all fields');
    }
    setAuthLoading(false);
  };

  const handleSignup = async (e) => {
    e && e.preventDefault();
    setAuthError('');
    if (!signupData.name || !signupData.email || !signupData.password) {
      setAuthError('Please fill in all fields');
      return;
    }
    if (signupData.password !== signupData.confirmPassword) {
      setAuthError('Passwords do not match');
      return;
    }
    setAuthLoading(true);
    await new Promise(r => setTimeout(r, 800));
    setUser({ name: signupData.name, email: signupData.email });
    setAuthState('app');
    setAuthLoading(false);
  };

  const handleLogout = () => {
    setUser(null);
    setAuthState('login');
    setFiles([]);
    setResults([]);
    setJsonAnalysis(null);
    setJsonInput('');
  };

  // ---------- File helpers ----------
  const handleFileSelect = async (e) => {
    const list = Array.from(e.target.files || []);
    for (const f of list) {
      await analyzeAndAddFile(f);
    }
  };

  const handleFileDrop = async (e) => {
    e.preventDefault();
    const list = Array.from(e.dataTransfer.files || []);
    for (const f of list) {
      await analyzeAndAddFile(f);
    }
  };

  const analyzeAndAddFile = async (file) => {
    // upload to backend /analyze_file
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/analyze_file`, { method: "POST", body: fd });
      const j = await res.json();
      if (!j.ok) {
        setResults(prev => [...prev, { name: file.name, status: "error", error: j.detail || "analyze failed" }]);
        return;
      }
      const entry = {
        file,
        filename: file.name,
        tmp_path: j.tmp_path,
        info: j.info,
        status: "pending",
        customFolder: null
      };
      setFiles(prev => [...prev, entry]);
    } catch (err) {
      setResults(prev => [...prev, { name: file.name, status: "error", error: String(err) }]);
    }
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  // ---------- Process / Save ----------
  const handleProcessAll = async () => {
    setProcessing(true);
    const out = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      // call /save_file with tmp_path and custom folder (if provided)
      const fd = new FormData();
      fd.append("tmp_path", f.tmp_path);
      if (f.customFolder) fd.append("custom_folder", f.customFolder);
      try {
        const res = await fetch(`${API_BASE}/save_file`, { method: "POST", body: fd });
        const j = await res.json();
        if (j.ok) {
          out.push({ name: f.filename, status: "saved", result: j.result });
        } else {
          out.push({ name: f.filename, status: "error", error: j.detail || "save failed" });
        }
      } catch (err) {
        out.push({ name: f.filename, status: "error", error: String(err) });
      }
    }

    // If JSON present and analyzed as valid, call ingest
    if (jsonInput && jsonAnalysis?.valid) {
      try {
        const res = await fetch(`${API_BASE}/ingest_json`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: jsonInput
        });
        const j = await res.json();
        if (j.ok) {
          out.push({ name: "JSON Batch", status: "saved", result: j.result });
        } else {
          out.push({ name: "JSON Batch", status: "error", error: j.detail || "ingest failed" });
        }
      } catch (err) {
        out.push({ name: "JSON Batch", status: "error", error: String(err) });
      }
    }

    setResults(out);
    setProcessing(false);
  };

  // ---------- JSON analyze ----------
  const analyzeJSON = async () => {
    if (!jsonInput) return;
    try {
      const res = await fetch(`${API_BASE}/analyze_json`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonInput
      });
      const j = await res.json();
      if (j.ok) {
        setJsonAnalysis(j.analysis);
      } else {
        setJsonAnalysis({ valid: false, error: j.detail || "analysis failed" });
      }
    } catch (err) {
      setJsonAnalysis({ valid: false, error: String(err) });
    }
  };

  // ---------- Icon helper ----------
  const FileTypeIcon = ({ type }) => {
    switch(type) {
      case 'image': return <Image className="w-5 h-5 text-purple-500" />;
      case 'video': return <Video className="w-5 h-5 text-blue-500" />;
      case 'json': return <Database className="w-5 h-5 text-green-500" />;
      case 'document': return <FileText className="w-5 h-5 text-orange-500" />;
      default: return <FileText className="w-5 h-5 text-gray-500" />;
    }
  };

  // ---------- UI screens (login/signup omitted for brevity) ----------
  if (authState === 'login') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <FolderTree className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-white mb-2">CortexVault</h1>
            <p className="text-purple-300">Intelligent File Organization</p>
          </div>

          <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-white/10 p-8 shadow-2xl">
            <h2 className="text-2xl font-bold text-white mb-6">Welcome Back</h2>

            {authError && (
              <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span className="text-sm">{authError}</span>
              </div>
            )}

            <form onSubmit={handleLogin} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                <input
                  type="email"
                  value={loginData.email}
                  onChange={(e) => setLoginData({...loginData, email: e.target.value})}
                  className="w-full px-4 py-3 bg-black/30 border border-white/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/50 transition-all"
                  placeholder="you@example.com"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Password</label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={loginData.password}
                    onChange={(e) => setLoginData({...loginData, password: e.target.value})}
                    className="w-full px-4 py-3 bg-black/30 border border-white/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/50 transition-all pr-12"
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={authLoading}
                className="w-full px-4 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 rounded-lg text-white font-semibold transition-all hover:shadow-lg hover:shadow-purple-500/50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {authLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Signing in...
                  </>
                ) : (
                  <>
                    <LogIn className="w-5 h-5" />
                    Sign In
                  </>
                )}
              </button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-gray-400 text-sm">
                Don't have an account?{' '}
                <button
                  onClick={() => { setAuthState('signup'); setAuthError(''); }}
                  className="text-purple-400 hover:text-purple-300 font-medium transition-colors"
                >
                  Sign up
                </button>
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (authState === 'signup') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <FolderTree className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-white mb-2">CortexVault</h1>
            <p className="text-purple-300">Intelligent File Organization</p>
          </div>

          <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-white/10 p-8 shadow-2xl">
            <h2 className="text-2xl font-bold text-white mb-6">Create Account</h2>

            {authError && (
              <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span className="text-sm">{authError}</span>
              </div>
            )}

            <form onSubmit={handleSignup} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Full Name</label>
                <input
                  type="text"
                  value={signupData.name}
                  onChange={(e) => setSignupData({...signupData, name: e.target.value})}
                  className="w-full px-4 py-3 bg-black/30 border border-white/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/50 transition-all"
                  placeholder="John Doe"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                <input
                  type="email"
                  value={signupData.email}
                  onChange={(e) => setSignupData({...signupData, email: e.target.value})}
                  className="w-full px-4 py-3 bg-black/30 border border-white/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/50 transition-all"
                  placeholder="you@example.com"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Password</label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={signupData.password}
                    onChange={(e) => setSignupData({...signupData, password: e.target.value})}
                    className="w-full px-4 py-3 bg-black/30 border border-white/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/50 transition-all pr-12"
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Confirm Password</label>
                <input
                  type={showPassword ? "text" : "password"}
                  value={signupData.confirmPassword}
                  onChange={(e) => setSignupData({...signupData, confirmPassword: e.target.value})}
                  className="w-full px-4 py-3 bg-black/30 border border-white/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/50 transition-all"
                  placeholder="••••••••"
                />
              </div>

              <button
                type="submit"
                disabled={authLoading}
                className="w-full px-4 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 rounded-lg text-white font-semibold transition-all hover:shadow-lg hover:shadow-purple-500/50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {authLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Creating account...
                  </>
                ) : (
                  <>
                    <UserPlus className="w-5 h-5" />
                    Create Account
                  </>
                )}
              </button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-gray-400 text-sm">
                Already have an account?{' '}
                <button
                  onClick={() => { setAuthState('login'); setAuthError(''); }}
                  className="text-purple-400 hover:text-purple-300 font-medium transition-colors"
                >
                  Sign in
                </button>
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ---------- Main App ----------
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="bg-black/30 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <FolderTree className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">CortexVault</h1>
                <p className="text-sm text-purple-300">Intelligent File Organization</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-white text-sm font-medium">{user?.name}</p>
                <p className="text-gray-400 text-xs">{user?.email}</p>
              </div>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white transition-colors"
              >
                Settings
              </button>
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 rounded-lg text-red-300 transition-colors"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {showSettings && (
          <div className="mb-6 p-6 bg-black/40 backdrop-blur-md rounded-xl border border-white/10">
            <h3 className="text-lg font-semibold text-white mb-4">Settings</h3>
            <div className="space-y-3">
              <label className="flex items-center gap-3 text-white cursor-pointer group">
                <input
                  type="checkbox"
                  checked={settings.showThumbnails}
                  onChange={(e) => setSettings({...settings, showThumbnails: e.target.checked})}
                  className="w-4 h-4 rounded accent-purple-500"
                />
                <span className="group-hover:text-purple-300 transition-colors">Show video thumbnails</span>
              </label>
              <label className="flex items-center gap-3 text-white cursor-pointer group">
                <input
                  type="checkbox"
                  checked={settings.showDocPreview}
                  onChange={(e) => setSettings({...settings, showDocPreview: e.target.checked})}
                  className="w-4 h-4 rounded accent-purple-500"
                />
                <span className="group-hover:text-purple-300 transition-colors">Show document preview</span>
              </label>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6">
            <div className="bg-black/40 backdrop-blur-md rounded-xl border border-white/10 p-6 shadow-xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Files
              </h2>

              <div
                onDrop={handleFileDrop}
                onDragOver={(e) => e.preventDefault()}
                className="border-2 border-dashed border-purple-500/50 rounded-lg p-8 text-center hover:border-purple-500 hover:bg-purple-500/10 transition-all cursor-pointer bg-purple-500/5"
              >
                <input
                  type="file"
                  multiple
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="w-12 h-12 text-purple-400 mx-auto mb-3" />
                  <p className="text-white font-medium mb-1">Drop files here or click to browse</p>
                  <p className="text-sm text-gray-400">Supports images, videos, documents, and JSON</p>
                </label>
              </div>

              {files.length > 0 && (
                <div className="mt-4 space-y-2 max-h-96 overflow-y-auto">
                  {files.map((file, index) => (
                    <div key={index} className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 transition-colors">
                      <FileTypeIcon type={file.info?.type || 'document'} />
                      <div className="flex-1 min-w-0">
                        <p className="text-white text-sm font-medium truncate">{file.filename}</p>
                        <p className="text-gray-400 text-xs">{file.info?.suggested_folder || '—'}</p>
                      </div>
                      <input
                        placeholder="Override folder (optional)"
                        value={file.customFolder || ''}
                        onChange={(e) => {
                          const cf = e.target.value;
                          setFiles(prev => prev.map((p, idx) => idx === index ? {...p, customFolder: cf} : p));
                        }}
                        className="bg-black/20 text-white px-2 py-1 rounded mr-2"
                      />
                      <button
                        onClick={() => removeFile(index)}
                        className="text-red-400 hover:text-red-300 hover:bg-red-500/10 p-1 rounded transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-black/40 backdrop-blur-md rounded-xl border border-white/10 p-6 shadow-xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Database className="w-5 h-5" />
                JSON Data
              </h2>

              <textarea
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                placeholder="Paste your JSON here (object, array, or multi-collection)..."
                className="w-full h-64 px-4 py-3 bg-black/30 border border-white/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/50 font-mono text-sm resize-none transition-all"
              />

              <div className="flex gap-3">
                <button
                  onClick={analyzeJSON}
                  disabled={!jsonInput}
                  className="mt-3 w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-all hover:shadow-lg hover:shadow-purple-500/50"
                >
                  Analyze JSON
                </button>
                <button
                  onClick={async () => {
                    if (!jsonInput) return;
                    try {
                      const res = await fetch(`${API_BASE}/ingest_json`, {
                        method: "POST", headers: {"Content-Type": "application/json"}, body: jsonInput
                      });
                      const j = await res.json();
                      setResults(prev => [...prev, {name: "JSON Ingest", result: j}]);
                    } catch (err) {
                      setResults(prev => [...prev, {name: "JSON Ingest", error: String(err)}]);
                    }
                  }}
                  disabled={!jsonInput}
                  className="mt-3 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-all"
                >
                  Ingest JSON Now
                </button>
              </div>

              {jsonAnalysis && (
                <div className="mt-4 p-4 bg-black/30 rounded-lg border border-white/10">
                  {jsonAnalysis.valid ? (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-green-400">
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">Valid JSON</span>
                      </div>
                      <div className="text-sm text-gray-300 space-y-1 pl-7">
                        <p>Structure: <span className="text-purple-400 font-mono">{jsonAnalysis.structure}</span></p>
                        <p>Collections: <span className="text-purple-400 font-mono">{jsonAnalysis.collections}</span></p>
                        <p>Total Records: <span className="text-purple-400 font-mono">{jsonAnalysis.totalRecords}</span></p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-start gap-2 text-red-400">
                      <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="font-medium">Invalid JSON</p>
                        <p className="text-sm text-gray-400 mt-1">{jsonAnalysis.error}</p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {(files.length > 0 || (jsonInput && jsonAnalysis?.valid)) && (
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleProcessAll}
              disabled={processing}
              className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 rounded-xl text-white font-semibold text-lg shadow-xl shadow-purple-500/50 transition-all transform hover:scale-105 hover:shadow-2xl hover:shadow-purple-500/60 disabled:scale-100 disabled:cursor-not-allowed disabled:shadow-none flex items-center gap-3"
            >
              {processing ? (
                <>
                  <Loader2 className="w-6 h-6 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <CheckCircle className="w-6 h-6" />
                  Process & Save All
                </>
              )}
            </button>
          </div>
        )}

        {results.length > 0 && (
          <div className="mt-8 bg-black/40 backdrop-blur-md rounded-xl border border-white/10 p-6 shadow-xl">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              Processing Complete
            </h2>

            <div className="space-y-2">
              {results.map((result, index) => (
                <div key={index} className="flex items-center gap-3 p-4 bg-green-500/10 hover:bg-green-500/20 border border-green-500/30 rounded-lg transition-colors">
                  <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium">{result.name}</p>
                    <pre className="text-sm text-gray-400 truncate">{JSON.stringify(result.result || result.error || result, null, 2)}</pre>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 mt-12">
        <div className="text-center text-gray-400 text-sm">
          <p className="font-medium">CortexVault - Intelligent File Organization System</p>
          <p className="mt-1">Powered by AI-driven categorization and smart storage</p>
        </div>
      </div>
    </div>
  );
};

export default CortexVault;
