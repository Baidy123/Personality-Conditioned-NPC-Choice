using System;
using System.Collections.Generic;

namespace Dissertation.Live
{
    // Mirrors the live server payloads (spec section 5, live mode). Arrays
    // only — JsonUtility cannot parse maps — and the override flag is spelled
    // "overridden" because "override" is a C# keyword. To change the wire
    // format, change the Python side (experiments/rq3/live_session.py) and
    // these DTOs together.

    [Serializable]
    public class LiveEvent
    {
        public string name;
        public bool active;
        public string force_npc;      // "" when the event has no force
    }

    [Serializable]
    public class LiveLocation
    {
        public string id;
        public bool unlocked;
        public List<LiveEvent> events;
    }

    [Serializable]
    public class LiveNpc
    {
        public int slot;
        public string name;
        public string policy;
        public List<float> ocean;     // [O, C, E, A, N] in [-1, 1]
        public string location;       // "" before the first step
    }

    [Serializable]
    public class LiveState
    {
        public string world;
        public int step_count;
        public List<LiveLocation> locations;
        public List<LiveNpc> npcs;
    }

    [Serializable]
    public class ProbEntry
    {
        public string id;
        public float p;
    }

    [Serializable]
    public class StepEntry
    {
        public int slot;
        public string name;
        public string location;
        public string action;
        public bool moved;
        public bool overridden;
        public List<ProbEntry> location_probs;   // empty on overridden steps
        public List<ProbEntry> action_probs;
    }

    [Serializable]
    public class StepResponse
    {
        public int step_count;
        public List<StepEntry> steps;
    }

    [Serializable]
    public class ErrorResponse
    {
        public string error;
    }

    // ------------------------------------------------------------- requests --

    [Serializable]
    public class EventToggleRequest
    {
        public string location;
        public string @event;         // serialised as "event"
        public bool active;
    }

    [Serializable]
    public class UnlockRequest
    {
        public string location;
        public bool unlocked;
    }

    // Serialises to the exact trait dict /npc expects; sending all five
    // fields makes every apply a full personality replace.
    [Serializable]
    public class Ocean
    {
        public float openness;
        public float conscientiousness;
        public float extraversion;
        public float agreeableness;
        public float neuroticism;
    }

    [Serializable]
    public class NpcEditRequest       // "slot" present -> edit
    {
        public int slot;
        public Ocean ocean;
    }

    [Serializable]
    public class NpcAddRequest        // no "slot" -> append
    {
        public string name;
        public Ocean ocean;
    }
}
