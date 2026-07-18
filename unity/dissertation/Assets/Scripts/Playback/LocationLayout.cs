using System.Collections.Generic;
using UnityEngine;

namespace Dissertation.Playback
{
    // Placeholder village layout on a fixed 16:9 orthographic view
    // (size 5 -> x in [-8.9, 8.9], y in [-5, 5]): quiet side left/top,
    // busy centre, wilds right, danger in the far corner. Edit here, then
    // rerun Dissertation > Build Playback Scene.
    public static class LocationLayout
    {
        public struct Entry
        {
            public Vector2 Position;
            public Color Color;

            public Entry(float x, float y, Color color)
            {
                Position = new Vector2(x, y);
                Color = color;
            }
        }

        public static readonly Vector2 NpcStart = new Vector2(0f, -3.6f);

        // NPC stands slightly below a block's centre so the name plate stays legible.
        public static readonly Vector2 NpcOffset = new Vector2(0f, -0.55f);

        public static readonly Dictionary<string, Entry> Entries =
            new Dictionary<string, Entry>
            {
                { "library",    new Entry(-6.3f,  3.0f, new Color(0.35f, 0.55f, 0.85f)) },
                { "chapel",     new Entry(-2.8f,  3.4f, new Color(0.75f, 0.70f, 0.90f)) },
                { "tavern",     new Entry(-3.8f, -1.4f, new Color(0.90f, 0.55f, 0.25f)) },
                { "market",     new Entry( 0.3f,  0.2f, new Color(0.92f, 0.80f, 0.35f)) },
                { "arena",      new Entry( 3.8f, -1.2f, new Color(0.80f, 0.35f, 0.30f)) },
                { "forest",     new Entry( 6.3f,  3.0f, new Color(0.35f, 0.65f, 0.35f)) },
                { "enemy_camp", new Entry( 7.0f, -3.8f, new Color(0.40f, 0.20f, 0.20f)) },
            };
    }
}
