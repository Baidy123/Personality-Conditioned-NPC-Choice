using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace Dissertation.Playback
{
    // Up to three NPC slots (A/B/C), each with its own sequence dropdown;
    // Continue steps every active NPC in lockstep. Hotkeys: Space = continue,
    // H = hide control bar, R = restart all. Sequences are generated
    // independently: NPCs share the map visually but do not perceive each
    // other, so multi-NPC playback is demo material, not a study stimulus.
    public class PlaybackController : MonoBehaviour
    {
        public NpcAgent[] npcs;
        public Dropdown[] dropdowns;
        public Dropdown countDropdown;   // how many slots are in play (1..npcs.Length)
        public GameObject[] slotRows;    // per-slot caption+dropdown rows, hidden when unused
        public Button continueButton;
        public Button restartButton;
        public Text statusText;
        public GameObject controlBar;
        public GameObject[] badges;   // A/B/C letters; hidden with the bar for clean frames

        Sequence[] sequences;
        PlaybackModel[] models;
        string[] files = new string[0];

        static string SlotName(int slot) => ((char)('A' + slot)).ToString();

        void Start()
        {
            var dir = Path.Combine(Application.streamingAssetsPath, "sequences");
            files = Directory.Exists(dir)
                ? Directory.GetFiles(dir, "*.json").OrderBy(f => f).ToArray()
                : new string[0];
            sequences = new Sequence[npcs.Length];
            models = new PlaybackModel[npcs.Length];
            var options = new List<string> { "(none)" };
            options.AddRange(files.Select(Path.GetFileNameWithoutExtension));
            for (var i = 0; i < dropdowns.Length; i++)
            {
                var slot = i;
                dropdowns[i].ClearOptions();
                dropdowns[i].AddOptions(options);
                dropdowns[i].onValueChanged.AddListener(_ => LoadSlot(slot));
                npcs[i].gameObject.SetActive(false);
            }
            countDropdown.ClearOptions();
            countDropdown.AddOptions(
                Enumerable.Range(1, npcs.Length).Select(n => n.ToString()).ToList());
            countDropdown.onValueChanged.AddListener(_ => ApplyCount());
            continueButton.onClick.AddListener(NextStep);
            restartButton.onClick.AddListener(ReloadAll);
            ApplyCount();                                   // default: 1 NPC
            if (files.Length > 0) dropdowns[0].value = 1;   // fires LoadSlot(0)
            else SetStatus("no sequences in StreamingAssets/sequences");
        }

        void ApplyCount()
        {
            var count = countDropdown.value + 1;
            for (var i = 0; i < slotRows.Length; i++)
            {
                var active = i < count;
                slotRows[i].SetActive(active);
                if (!active)
                {
                    // clear directly — relying on the dropdown's value-changed
                    // event here left the NPC alive when the event didn't fire
                    dropdowns[i].SetValueWithoutNotify(0);
                    ClearSlot(i);
                }
            }
        }

        void ClearSlot(int slot)
        {
            models[slot] = null;
            sequences[slot] = null;
            npcs[slot].gameObject.SetActive(false);
        }

        void LoadSlot(int slot)
        {
            ClearSlot(slot);
            var sel = dropdowns[slot].value;                // 0 = "(none)"
            if (sel == 0)
            {
                SetStatus($"{SlotName(slot)} cleared");
                return;
            }
            Sequence loaded;
            try
            {
                loaded = Sequence.FromJson(File.ReadAllText(files[sel - 1]));
                foreach (var step in loaded.steps)
                    if (!LocationLayout.Entries.ContainsKey(step.location))
                        throw new System.InvalidOperationException(
                            $"location {step.location} has no layout entry");
            }
            catch (System.Exception e)
            {
                SetStatus($"load error ({SlotName(slot)}): {e.Message}");
                return;
            }
            sequences[slot] = loaded;
            models[slot] = new PlaybackModel(loaded.steps);
            npcs[slot].gameObject.SetActive(true);
            npcs[slot].Teleport(LocationLayout.NpcStart + LocationLayout.SlotOffsets[slot]);
            SetStatus($"{SlotName(slot)} = {loaded.meta.sequence_id} "
                      + $"({loaded.steps.Count} steps) - Continue to start");
        }

        void ReloadAll()
        {
            for (var i = 0; i < dropdowns.Length; i++)
                if (dropdowns[i].value > 0) LoadSlot(i);
        }

        void NextStep()
        {
            // all-or-nothing: advancing only the slots that finished walking
            // would let lockstep cycles drift apart permanently
            for (var i = 0; i < npcs.Length; i++)
                if (models[i] != null && npcs[i].IsWalking)
                {
                    SetStatus("walking...");
                    return;
                }
            var parts = new List<string>();
            for (var i = 0; i < npcs.Length; i++)
            {
                if (models[i] == null) continue;
                var step = models[i].Advance();
                if (step == null)
                {
                    npcs[i].StopAll();
                    parts.Add($"{SlotName(i)} finished");
                    continue;
                }
                var pos = LocationLayout.Entries[step.location].Position
                          + LocationLayout.NpcOffset + LocationLayout.SlotOffsets[i];
                var slot = i;               // avoid loop-variable capture
                var action = step.action;
                if (step.moved) npcs[i].WalkTo(pos, () => npcs[slot].PerformAction(action));
                else npcs[i].PerformAction(action);
                parts.Add($"{SlotName(i)} {step.cycle}/{sequences[i].steps.Count} "
                          + $"{step.location}/{step.action}");
            }
            if (parts.Count > 0) SetStatus(string.Join("   ", parts));
        }

        void Update()
        {
            if (Input.GetKeyDown(KeyCode.Space)) NextStep();
            if (Input.GetKeyDown(KeyCode.H))
            {
                var show = !controlBar.activeSelf;
                controlBar.SetActive(show);
                if (badges != null)
                    foreach (var badge in badges) badge.SetActive(show);
            }
            if (Input.GetKeyDown(KeyCode.R)) ReloadAll();
        }

        void SetStatus(string msg)
        {
            if (statusText != null) statusText.text = msg;
        }

        // Mode switching (see Dissertation.Live.ModeSwitcher): hide this
        // mode's NPCs without losing which slots are loaded.
        public void SetNpcsVisible(bool visible)
        {
            for (var i = 0; i < npcs.Length; i++)
            {
                var loaded = models != null && i < models.Length && models[i] != null;
                npcs[i].gameObject.SetActive(visible && loaded);
            }
        }
    }
}
