using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace Dissertation.Playback
{
    // Orchestrates one sequence: dropdown -> load JSON -> step on Continue or
    // auto-advance. Hotkeys: Space = continue, H = hide control bar, R = restart.
    // Recording mode = Auto ON + control bar hidden: the frame then contains
    // only the scene, location name plates, and the NPC action label.
    public class PlaybackController : MonoBehaviour
    {
        public NpcAgent npc;
        public Dropdown sequenceDropdown;
        public Button continueButton;
        public Button restartButton;
        public Toggle autoToggle;
        public Text statusText;
        public GameObject controlBar;
        public float autoIntervalSeconds = 4f;

        Sequence sequence;
        PlaybackModel model;
        bool performing;          // NPC currently looping an action
        float performTimer;
        string[] files = new string[0];

        void Start()
        {
            var dir = Path.Combine(Application.streamingAssetsPath, "sequences");
            files = Directory.Exists(dir)
                ? Directory.GetFiles(dir, "*.json").OrderBy(f => f).ToArray()
                : new string[0];
            sequenceDropdown.ClearOptions();
            sequenceDropdown.AddOptions(files.Select(Path.GetFileNameWithoutExtension).ToList());
            sequenceDropdown.onValueChanged.AddListener(_ => LoadSelected());
            continueButton.onClick.AddListener(NextStep);
            restartButton.onClick.AddListener(LoadSelected);
            if (files.Length > 0) LoadSelected();
            else SetStatus("no sequences in StreamingAssets/sequences");
        }

        void LoadSelected()
        {
            performing = false;
            model = null;
            sequence = null;
            var path = files[sequenceDropdown.value];
            Sequence loaded;
            try
            {
                loaded = Sequence.FromJson(File.ReadAllText(path));
                foreach (var step in loaded.steps)
                    if (!LocationLayout.Entries.ContainsKey(step.location))
                        throw new System.InvalidOperationException(
                            $"location {step.location} has no layout entry");
            }
            catch (System.Exception e)
            {
                SetStatus("load error: " + e.Message);
                return;
            }
            sequence = loaded;
            model = new PlaybackModel(sequence.steps);
            npc.Teleport(LocationLayout.NpcStart);
            SetStatus($"{sequence.meta.sequence_id} ready ({sequence.steps.Count} steps)"
                      + " - Continue to start");
        }

        void NextStep()
        {
            if (model == null || npc.IsWalking) return;
            var step = model.Advance();
            if (step == null)
            {
                npc.StopAll();
                performing = false;
                SetStatus($"{sequence.meta.sequence_id} finished - R to restart");
                return;
            }
            performing = false;
            var pos = LocationLayout.Entries[step.location].Position + LocationLayout.NpcOffset;
            if (step.moved) npc.WalkTo(pos, () => BeginPerform(step));
            else BeginPerform(step);
            SetStatus($"step {step.cycle}/{sequence.steps.Count}: {step.location} / {step.action}");
        }

        void BeginPerform(SequenceStep step)
        {
            npc.PerformAction(step.action);
            performing = true;
            performTimer = 0f;
        }

        void Update()
        {
            if (Input.GetKeyDown(KeyCode.Space)) NextStep();
            if (Input.GetKeyDown(KeyCode.H)) controlBar.SetActive(!controlBar.activeSelf);
            if (Input.GetKeyDown(KeyCode.R)) LoadSelected();
            if (performing && autoToggle.isOn)
            {
                performTimer += Time.deltaTime;
                if (performTimer >= autoIntervalSeconds)
                {
                    performing = false;
                    NextStep();
                }
            }
        }

        void SetStatus(string msg)
        {
            if (statusText != null) statusText.text = msg;
        }
    }
}
