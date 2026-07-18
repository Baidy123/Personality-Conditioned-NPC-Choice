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
        public SpriteRenderer labelBox;       // dark frame behind the text

        const int BaseLabelOrder = 10;        // hover lifts to HoverLabelOrder
        const int HoverLabelOrder = 60;
        MeshRenderer labelRenderer;

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
            FitLabelBox();
        }

        public void StopAll()
        {
            if (performing) transform.position = basePosition;
            walking = false;
            performing = false;
            if (actionLabel != null) actionLabel.text = "";
            FitLabelBox();
        }

        // size the frame to the current text (label lossyScale ~ 1, so the
        // renderer's world bounds map straight onto the box's local scale)
        void FitLabelBox()
        {
            if (labelBox == null || actionLabel == null) return;
            var hasText = !string.IsNullOrEmpty(actionLabel.text);
            labelBox.enabled = hasText;
            if (!hasText) return;
            var size = actionLabel.GetComponent<MeshRenderer>().bounds.size;
            labelBox.transform.localScale =
                new Vector3(size.x + 0.35f, size.y + 0.18f, 1f);
        }

        // hover brings this NPC's label (and frame) to the top layer, so
        // co-located NPCs' overlapping labels are readable one at a time
        void OnMouseEnter() => SetLabelOnTop(true);
        void OnMouseExit() => SetLabelOnTop(false);

        public void SetLabelOnTop(bool onTop)
        {
            var order = onTop ? HoverLabelOrder : BaseLabelOrder;
            if (labelRenderer == null && actionLabel != null)
                labelRenderer = actionLabel.GetComponent<MeshRenderer>();
            if (labelRenderer != null) labelRenderer.sortingOrder = order + 1;
            if (labelBox != null) labelBox.sortingOrder = order;
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
