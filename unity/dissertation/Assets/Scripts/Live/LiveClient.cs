using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace Dissertation.Live
{
    // Thin coroutine wrappers over UnityWebRequest against the local brain
    // server (experiments/rq3/live_server.py). Localhost inter-process
    // channel only — nothing here ever leaves the machine.
    public class LiveClient : MonoBehaviour
    {
        public string baseUrl = "http://127.0.0.1:8973";

        public void Get(string path, Action<string> onOk, Action<string> onError)
        {
            StartCoroutine(Run(UnityWebRequest.Get(baseUrl + path), onOk, onError));
        }

        public void Post(string path, string jsonBody,
                         Action<string> onOk, Action<string> onError)
        {
            var req = new UnityWebRequest(baseUrl + path, "POST");
            var payload = System.Text.Encoding.UTF8.GetBytes(
                string.IsNullOrEmpty(jsonBody) ? "{}" : jsonBody);
            req.uploadHandler = new UploadHandlerRaw(payload);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            StartCoroutine(Run(req, onOk, onError));
        }

        static IEnumerator Run(UnityWebRequest req,
                               Action<string> onOk, Action<string> onError)
        {
            using (req)
            {
                yield return req.SendWebRequest();
                var body = req.downloadHandler != null ? req.downloadHandler.text : "";
                if (req.result == UnityWebRequest.Result.Success)
                {
                    onOk?.Invoke(body);
                    yield break;
                }
                // 400s carry {"error": ...}; connection failures carry req.error
                var msg = req.error;
                if (!string.IsNullOrEmpty(body))
                {
                    try
                    {
                        var parsed = JsonUtility.FromJson<ErrorResponse>(body);
                        if (parsed != null && !string.IsNullOrEmpty(parsed.error))
                            msg = parsed.error;
                    }
                    catch (Exception) { /* keep req.error */ }
                }
                onError?.Invoke(msg);
            }
        }
    }
}
