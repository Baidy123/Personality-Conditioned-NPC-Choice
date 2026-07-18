using System;
using System.Collections.Generic;
using UnityEngine;

namespace Dissertation.Playback
{
    // The Inspector-facing art interface: a persistent asset
    // (Assets/ArtSet.asset, auto-created by the scene builder) with one
    // sprite slot per visual. Drag imported images into the slots and rerun
    // Dissertation > Build Playback Scene. Empty slot -> the builder falls
    // back to Assets/Art/<name>.png, then to the colour placeholder.
    // Unlike edits made directly on scene objects, this asset survives
    // every rebuild.
    [CreateAssetMenu(fileName = "ArtSet", menuName = "Dissertation/Art Set")]
    public class ArtSet : ScriptableObject
    {
        [Serializable]
        public class LocationArt
        {
            public string id;         // must match a LocationLayout entry
            public Sprite sprite;
        }

        public Sprite ground;
        public Sprite npc;            // per-slot tints still apply: prefer light art
        public List<LocationArt> locations = new List<LocationArt>();

        public Sprite Location(string id)
        {
            foreach (var entry in locations)
                if (entry.id == id) return entry.sprite;
            return null;
        }
    }
}
