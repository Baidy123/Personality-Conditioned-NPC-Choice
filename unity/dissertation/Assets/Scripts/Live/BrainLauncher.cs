using System;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Dissertation.Live
{
    // Starts/stops the Python brain as a hidden child process, so live mode
    // is one click with no terminal. The brain's console output lands in the
    // Unity Console with a [brain] prefix.
    //
    // Dev default: run `python -m experiments.rq3.live_server ...` from the
    // repo's code/ directory (derived from the project path). Packaged
    // build: point `executable` at the bundled npc_brain.exe (PyInstaller)
    // and set `arguments` to just `--config <file>` — nothing else changes.
    public class BrainLauncher : MonoBehaviour
    {
        public string executable = "python";
        public string arguments =
            "-m experiments.rq3.live_server --config experiments/rq3/live_config_demo.json";
        [Tooltip("Blank = auto: the repo's code/ folder in the editor, the exe folder in builds")]
        public string workingDirectory = "";

        Process process;

        public bool IsRunning => process != null && !process.HasExited;

        // Returns true if a brain process is (already) running or was started.
        // A false return is not fatal: Connect's retry loop can still reach a
        // manually started server on the same port.
        public bool StartBrain()
        {
            if (IsRunning) return true;
            var workDir = string.IsNullOrEmpty(workingDirectory)
                ? DefaultWorkingDirectory()
                : workingDirectory;
            try
            {
                process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = executable,
                        Arguments = arguments,
                        WorkingDirectory = workDir,
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                    },
                    EnableRaisingEvents = true,
                };
                process.OutputDataReceived += (_, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data)) Debug.Log("[brain] " + e.Data);
                };
                process.ErrorDataReceived += (_, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data)) Debug.LogWarning("[brain] " + e.Data);
                };
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                Debug.Log($"[brain] started: {executable} {arguments} (cwd {workDir})");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[brain] failed to start '{executable}': {e.Message}");
                process = null;
                return false;
            }
        }

        public void StopBrain()
        {
            if (process == null) return;
            try
            {
                if (!process.HasExited) process.Kill();
                process.Dispose();
            }
            catch (Exception) { /* already gone */ }
            process = null;
        }

        static string DefaultWorkingDirectory()
        {
#if UNITY_EDITOR
            // Assets -> dissertation -> unity -> code
            return Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "..", ".."));
#else
            return Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
#endif
        }

        void OnApplicationQuit() => StopBrain();
        void OnDestroy() => StopBrain();
    }
}
