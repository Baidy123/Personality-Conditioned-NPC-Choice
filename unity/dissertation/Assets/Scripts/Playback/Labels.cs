using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Dissertation.Playback
{
    // Bilingual display names for location/action ids, loaded once from
    // StreamingAssets/labels.json (generated from data/rq3_labels_zh.json +
    // rq3_labels_en.json — edit those and regenerate, not this file's data).
    // Chinese is the participant-facing line; English sits beneath it for the
    // record. Unknown ids fall back to the raw id so a missing entry is
    // visible rather than silent.
    [Serializable]
    public class LabelEntry { public string id; public string zh; public string en; }

    [Serializable]
    public class LabelFile
    {
        public List<LabelEntry> locations;
        public List<LabelEntry> actions;
    }

    public static class Labels
    {
        static Dictionary<string, string> loc, act;

        static void Load()
        {
            loc = new Dictionary<string, string>();
            act = new Dictionary<string, string>();
            var path = Path.Combine(Application.streamingAssetsPath, "labels.json");
            if (!File.Exists(path))
            {
                Debug.LogWarning($"Labels: {path} not found; ids will be shown raw.");
                return;
            }
            var f = JsonUtility.FromJson<LabelFile>(File.ReadAllText(path));
            foreach (var e in f.locations) loc[e.id] = e.zh + "\n" + e.en;
            foreach (var e in f.actions) act[e.id] = e.zh + "\n" + e.en;
        }

        public static string Location(string id)
        {
            if (loc == null) Load();
            return loc.TryGetValue(id, out var s) ? s : id;
        }

        public static string Action(string id)
        {
            if (act == null) Load();
            return act.TryGetValue(id, out var s) ? s : id;
        }
    }
}
