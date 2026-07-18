using System;
using UnityEngine;

namespace Dissertation.Playback
{
    // Moves the NPC dot in a straight line between locations and loops a
    // small bob while an action is performed; the label above shows the
    // action id (the only per-step text participants see besides name plates).
    public class NpcAgent : MonoBehaviour
    {
        public float moveSpeed = 3f;          // world units / second
        public TextMesh actionLabel;

        Vector2 target;
        Action onArrive;
        bool walking;
        bool performing;
        Vector3 basePosition;
        float bobT;

        public bool IsWalking => walking;

        public void Teleport(Vector2 pos)
        {
            StopAll();
            transform.position = pos;
        }

        public void WalkTo(Vector2 pos, Action arrived)
        {
            StopAll();
            target = pos;
            onArrive = arrived;
            walking = true;
        }

        public void PerformAction(string actionId)
        {
            // consecutive same-location actions: drop any bob offset before
            // re-capturing basePosition, or the NPC creeps upward step by step
            if (performing) transform.position = basePosition;
            walking = false;
            performing = true;
            basePosition = transform.position;
            bobT = 0f;
            actionLabel.text = actionId;
        }

        public void StopAll()
        {
            if (performing) transform.position = basePosition;
            walking = false;
            performing = false;
            if (actionLabel != null) actionLabel.text = "";
        }

        void Update()
        {
            if (walking)
            {
                transform.position = Vector2.MoveTowards(
                    transform.position, target, moveSpeed * Time.deltaTime);
                if ((Vector2)transform.position == target)
                {
                    walking = false;
                    onArrive?.Invoke();
                }
            }
            else if (performing)
            {
                bobT += Time.deltaTime;
                transform.position = basePosition
                    + new Vector3(0f, Mathf.Abs(Mathf.Sin(bobT * 4f)) * 0.15f, 0f);
            }
        }
    }
}
