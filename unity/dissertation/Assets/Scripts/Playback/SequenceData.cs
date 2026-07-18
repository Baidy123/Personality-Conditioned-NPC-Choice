using System;
using System.Collections.Generic;
using UnityEngine;

namespace Dissertation.Playback
{
    // Parsed view of one sequence JSON produced by experiments/rq3.
    // Condition fields (policy, checkpoint, personality, seed) are deliberately
    // not parsed, so nothing condition-revealing can reach the UI; the
    // *_probs research-archive fields are ignored by JsonUtility.
    [Serializable]
    public class SequenceMeta
    {
        public string sequence_id;
        public string world;
        public int n_cycles;
    }

    [Serializable]
    public class SequenceStep
    {
        public int cycle;
        public string location;
        public string action;
        public bool moved;
    }

    [Serializable]
    public class Sequence
    {
        public SequenceMeta meta;
        public List<SequenceStep> steps;

        public static Sequence FromJson(string json)
        {
            var seq = JsonUtility.FromJson<Sequence>(json);
            seq.Validate();
            return seq;
        }

        // Player-side mirror of the Python export gate: reject rather than
        // play a malformed file.
        public void Validate()
        {
            if (meta == null || steps == null || steps.Count == 0)
                throw new InvalidOperationException("sequence: missing meta or steps");
            if (steps.Count != meta.n_cycles)
                throw new InvalidOperationException(
                    $"{meta.sequence_id}: {steps.Count} steps != n_cycles {meta.n_cycles}");
            if (!steps[0].moved)
                throw new InvalidOperationException(
                    $"{meta.sequence_id}: first step must have moved=true");
            string prev = null;
            foreach (var s in steps)
            {
                if (string.IsNullOrEmpty(s.location) || string.IsNullOrEmpty(s.action))
                    throw new InvalidOperationException(
                        $"{meta.sequence_id} cycle {s.cycle}: empty location/action");
                if (s.moved != (s.location != prev))
                    throw new InvalidOperationException(
                        $"{meta.sequence_id} cycle {s.cycle}: moved={s.moved} inconsistent "
                        + $"with previous location {prev ?? "none"}");
                prev = s.location;
            }
        }
    }
}
