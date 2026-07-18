using System.Collections.Generic;

namespace Dissertation.Playback
{
    // Pure step cursor over a validated sequence; all Unity-side pacing
    // (walking, labels, auto-advance) hangs off this.
    public class PlaybackModel
    {
        readonly IReadOnlyList<SequenceStep> steps;

        public PlaybackModel(IReadOnlyList<SequenceStep> steps) => this.steps = steps;

        public int Index { get; private set; } = -1;   // -1 = before first step
        public bool Finished => Index >= steps.Count - 1;
        public SequenceStep Current => Index < 0 ? null : steps[Index];

        // Returns the next step, or null when the sequence is exhausted.
        public SequenceStep Advance()
        {
            if (Finished) return null;
            Index++;
            return steps[Index];
        }
    }
}
