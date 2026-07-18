using System.Collections;
using System.Collections.Generic;
using Dissertation.Playback;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace Dissertation.Live
{
    // Live demo frontend: the Python brain owns all state; this class only
    // renders it and forwards clicks. Connect -> /state spawns NPC dots and
    // builds the two side panels; Step/Space -> /step animates the returned
    // decisions; the event panel posts /event; the personality panel posts
    // /npc; R -> /reset. Every request error lands in the status line.
    public class LiveController : MonoBehaviour
    {
        public LiveClient client;
        public BrainLauncher launcher;         // optional: auto-starts the brain
        public NpcAgent npcPrototype;          // inactive template in the scene
        public Button connectButton;
        public Button stepButton;
        public Button resetButton;
        public Text statusText;
        public GameObject controlBar;
        public RectTransform eventPanelContent;   // rows rebuilt from /state
        public GameObject eventPanel;
        public GameObject personalityPanel;
        public Button personalityToggleButton;
        public Dropdown policyDropdown;   // global model switch (catalog from /state)
        public GameObject logPanel;
        public Text logText;
        public Button logToggleButton;
        public Dropdown npcDropdown;
        public Slider[] traitSliders;             // O, C, E, A, N
        public Text[] traitValueLabels;
        public Button applyButton;
        public InputField newNpcName;
        public Button addNpcButton;

        static readonly Color[] Palette =
        {
            new Color(0.13f, 0.13f, 0.16f),   // near-black
            new Color(0.30f, 0.12f, 0.35f),   // dark purple
            new Color(0.05f, 0.28f, 0.32f),   // dark teal
            new Color(0.35f, 0.25f, 0.05f),   // olive
            new Color(0.40f, 0.10f, 0.10f),   // maroon
            new Color(0.10f, 0.15f, 0.40f),   // navy
        };

        LiveState state;
        readonly List<NpcAgent> agents = new List<NpcAgent>();
        readonly List<GameObject> spawned = new List<GameObject>();
        readonly List<string> logHistory = new List<string>();
        const int LogTailLines = 22;
        bool busy;

        void Start()
        {
            connectButton.onClick.AddListener(Connect);
            stepButton.onClick.AddListener(Step);
            resetButton.onClick.AddListener(ResetSession);
            applyButton.onClick.AddListener(ApplyPersonality);
            addNpcButton.onClick.AddListener(AddNpc);
            personalityToggleButton.onClick.AddListener(
                () => TogglePanel(personalityPanel, logPanel));
            logToggleButton.onClick.AddListener(
                () => TogglePanel(logPanel, personalityPanel));
            policyDropdown.onValueChanged.AddListener(_ => OnPolicyPicked());
            npcDropdown.onValueChanged.AddListener(_ => LoadSlidersFromSelected());
            for (var i = 0; i < traitSliders.Length; i++)
            {
                var idx = i;
                traitSliders[i].onValueChanged.AddListener(
                    v => traitValueLabels[idx].text = v.ToString("+0.00;-0.00"));
            }
            SetStatus("Live mode - start the brain server, then Connect");
        }

        // ---------------------------------------------------------- requests --
        void Connect()
        {
            busy = false;                  // clear a flag stranded by a hang
            // auto-start the brain if a launcher is wired; even if that fails,
            // the retry loop can still reach a manually started server
            if (launcher != null && !launcher.IsRunning)
            {
                SetStatus("starting brain...");
                launcher.StartBrain();
            }
            StartCoroutine(ConnectRoutine());
        }

        IEnumerator ConnectRoutine()
        {
            const int attempts = 12;               // ~5 s of patience for boot
            for (var i = 0; i < attempts; i++)
            {
                if (!enabled) yield break;         // mode switched mid-connect
                var done = false;
                var success = false;
                client.Get("/state", body =>
                {
                    if (!isActiveAndEnabled) { done = true; return; }
                    state = JsonUtility.FromJson<LiveState>(body);
                    RebuildAll();
                    var names = new List<string>();
                    foreach (var npc in state.npcs) names.Add(npc.name);
                    Log($"connected: {string.Join(", ", names)}");
                    SetStatus($"connected - {state.npcs.Count} NPCs, "
                              + $"world {state.world}");
                    done = true;
                    success = true;
                }, _ => done = true);
                yield return new WaitUntil(() => done);
                if (success) yield break;
                yield return new WaitForSeconds(0.4f);
            }
            SetStatus("connect failed - check the Console for [brain] errors");
        }

        void Step()
        {
            if (state == null || busy) return;
            foreach (var agent in agents)
                if (agent.IsWalking)
                {
                    SetStatus("walking...");
                    return;
                }
            busy = true;
            client.Post("/step", null, body =>
            {
                busy = false;
                if (!isActiveAndEnabled) return;   // mode switched mid-request
                Animate(JsonUtility.FromJson<StepResponse>(body));
            }, err =>
            {
                busy = false;
                if (!isActiveAndEnabled) return;
                Log($"ERROR step: {err}");
                SetStatus($"step error: {err}");
            });
        }

        void ResetSession()
        {
            if (state == null) return;
            client.Post("/reset", null, body =>
            {
                if (!isActiveAndEnabled) return;   // mode switched mid-request
                state = JsonUtility.FromJson<LiveState>(body);
                RebuildAll();
                Log("reset: authored world and startup roster restored");
                SetStatus("reset - authored world and startup roster restored");
            }, err =>
            {
                Log($"ERROR reset: {err}");
                SetStatus($"reset error: {err}");
            });
        }

        void PostEvent(string json, string description)
        {
            client.Post("/event", json, body =>
            {
                if (!isActiveAndEnabled) return;   // mode switched mid-request
                state = JsonUtility.FromJson<LiveState>(body);
                RebuildEventPanel();     // NPCs unaffected: changes apply next step
                Log(description);
                SetStatus($"{description} - takes effect at the next step");
            }, err =>
            {
                Log($"ERROR {description}: {err}");
                SetStatus($"event error: {err}");
            });
        }

        void PostNpc(string json, string description)
        {
            client.Post("/npc", json, body =>
            {
                if (!isActiveAndEnabled) return;   // mode switched mid-request
                state = JsonUtility.FromJson<LiveState>(body);
                RebuildNpcs();
                RebuildNpcDropdown();
                RebuildPolicyDropdown();   // an added NPC can make it "mixed"
                Log(description);
                SetStatus($"{description} - takes effect at the next step");
            }, err =>
            {
                Log($"ERROR {description}: {err}");
                SetStatus($"npc error: {err}");
            });
        }

        // --------------------------------------------------------- personality --
        void LoadSlidersFromSelected()
        {
            if (state == null || state.npcs.Count == 0) return;
            var npc = state.npcs[Mathf.Clamp(npcDropdown.value, 0, state.npcs.Count - 1)];
            for (var i = 0; i < traitSliders.Length && i < npc.ocean.Count; i++)
            {
                traitSliders[i].SetValueWithoutNotify(npc.ocean[i]);
                traitValueLabels[i].text = npc.ocean[i].ToString("+0.00;-0.00");
            }
        }

        Ocean OceanFromSliders() => new Ocean
        {
            openness = traitSliders[0].value,
            conscientiousness = traitSliders[1].value,
            extraversion = traitSliders[2].value,
            agreeableness = traitSliders[3].value,
            neuroticism = traitSliders[4].value,
        };

        void ApplyPersonality()
        {
            if (state == null || state.npcs.Count == 0) return;
            var npc = state.npcs[Mathf.Clamp(npcDropdown.value, 0, state.npcs.Count - 1)];
            PostNpc(JsonUtility.ToJson(new NpcEditRequest
            {
                slot = npc.slot,
                ocean = OceanFromSliders(),
            }), $"personality applied to {npc.name}");
        }

        void AddNpc()
        {
            if (state == null) return;
            var name = newNpcName.text.Trim();
            if (name.Length == 0)
            {
                SetStatus("give the new NPC a name first");
                return;
            }
            PostNpc(JsonUtility.ToJson(new NpcAddRequest
            {
                name = name,
                ocean = OceanFromSliders(),
            }), $"added NPC {name}");
        }

        // ------------------------------------------------------------- render --
        void RebuildAll()
        {
            RebuildNpcs();
            RebuildEventPanel();
            RebuildNpcDropdown();
            RebuildPolicyDropdown();
        }

        void RebuildPolicyDropdown()
        {
            var options = new List<string>(state.policies);
            if (state.active_policy == "mixed") options.Insert(0, "mixed");
            policyDropdown.ClearOptions();
            policyDropdown.AddOptions(options);
            policyDropdown.SetValueWithoutNotify(
                Mathf.Max(0, options.IndexOf(state.active_policy)));
            policyDropdown.RefreshShownValue();
        }

        void OnPolicyPicked()
        {
            if (state == null) return;
            var label = policyDropdown.options[policyDropdown.value].text;
            if (label == "mixed" || label == state.active_policy) return;
            client.Post("/policy",
                JsonUtility.ToJson(new PolicyRequest { policy = label }), body =>
            {
                if (!isActiveAndEnabled) return;
                state = JsonUtility.FromJson<LiveState>(body);
                RebuildPolicyDropdown();
                Log($"policy -> {label} (all NPCs)");
                SetStatus($"policy -> {label} - takes effect at the next step");
            }, err =>
            {
                if (!isActiveAndEnabled) return;
                Log($"ERROR policy: {err}");
                SetStatus($"policy error: {err}");
            });
        }

        void RebuildNpcs()
        {
            foreach (var go in spawned) Destroy(go);
            spawned.Clear();
            agents.Clear();
            for (var i = 0; i < state.npcs.Count; i++)
            {
                var npc = state.npcs[i];
                var go = Instantiate(npcPrototype.gameObject);
                go.name = "LiveNPC_" + npc.name;
                go.SetActive(true);
                go.GetComponent<SpriteRenderer>().color = Palette[i % Palette.Length];
                var agent = go.GetComponent<NpcAgent>();

                var nameLabel = go.transform.Find("NameLabel");
                if (nameLabel != null)
                    nameLabel.GetComponent<TextMesh>().text = npc.name;

                // per-slot label height plane, same anti-overlap scheme as playback
                var offset = LocationLayout.SlotOffsets[
                    i % LocationLayout.SlotOffsets.Length];
                var plane = LocationLayout.LabelPlanes[
                    i % LocationLayout.LabelPlanes.Length];
                agent.actionLabel.transform.localPosition = new Vector3(
                    0f, (plane - offset.y) / go.transform.localScale.x, 0f);

                var pos = npc.location != ""
                          && LocationLayout.Entries.ContainsKey(npc.location)
                    ? LocationLayout.Entries[npc.location].Position
                      + LocationLayout.NpcOffset + offset
                    : LocationLayout.NpcStart + offset;
                agent.Teleport(pos);

                spawned.Add(go);
                agents.Add(agent);
            }
        }

        void RebuildEventPanel()
        {
            foreach (Transform child in eventPanelContent)
                Destroy(child.gameObject);
            foreach (var loc in state.locations)
            {
                AddPanelHeader(loc.id + (loc.unlocked ? "" : "  [locked]"));
                var locId = loc.id;
                var unlocked = loc.unlocked;
                AddPanelButton(unlocked ? "lock" : "unlock",
                    () => PostEvent(JsonUtility.ToJson(new UnlockRequest
                    {
                        location = locId,
                        unlocked = !unlocked,
                    }), $"{(unlocked ? "locked" : "unlocked")} {locId}"));
                foreach (var ev in loc.events)
                {
                    var evName = ev.name;
                    var active = ev.active;
                    var marker = string.IsNullOrEmpty(ev.force_npc) ? "" : "! ";
                    AddPanelButton($"{marker}{evName}: {(active ? "ON" : "OFF")}",
                        () => PostEvent(JsonUtility.ToJson(new EventToggleRequest
                        {
                            location = locId,
                            @event = evName,
                            active = !active,
                        }), $"event {evName} -> {(active ? "OFF" : "ON")} @ {locId}"));
                }
            }
        }

        void RebuildNpcDropdown()
        {
            var names = new List<string>();
            foreach (var npc in state.npcs) names.Add(npc.name);
            var keep = Mathf.Clamp(npcDropdown.value, 0, Mathf.Max(0, names.Count - 1));
            npcDropdown.ClearOptions();
            npcDropdown.AddOptions(names);
            npcDropdown.SetValueWithoutNotify(keep);
            npcDropdown.RefreshShownValue();
            LoadSlidersFromSelected();
        }

        void Animate(StepResponse resp)
        {
            var parts = new List<string>();
            foreach (var s in resp.steps)
            {
                if (s.slot < 0 || s.slot >= agents.Count) continue;
                if (!LocationLayout.Entries.ContainsKey(s.location))
                {
                    SetStatus($"no layout entry for {s.location}");
                    continue;
                }
                var agent = agents[s.slot];
                var offset = LocationLayout.SlotOffsets[
                    s.slot % LocationLayout.SlotOffsets.Length];
                var pos = LocationLayout.Entries[s.location].Position
                          + LocationLayout.NpcOffset + offset;
                var label = s.action + (s.overridden ? " (forced)" : "");
                if (s.moved) agent.WalkTo(pos, () => agent.PerformAction(label));
                else agent.PerformAction(label);
                parts.Add($"{s.name} {s.location}/{s.action}"
                          + (s.overridden ? " (forced)" : ""));
            }
            var line = $"step {resp.step_count}: {string.Join("  |  ", parts)}";
            Log(line);
            SetStatus(line);
        }

        // -------------------------------------------------------------- misc --
        public void SetNpcsVisible(bool visible)
        {
            foreach (var go in spawned)
                if (go != null) go.SetActive(visible);
        }

        void Update()
        {
            // typing a new NPC's name must not fire hotkeys ("r" = Reset!)
            if (newNpcName != null && newNpcName.isFocused) return;
            if (Input.GetKeyDown(KeyCode.Space)) Step();
            if (Input.GetKeyDown(KeyCode.R)) ResetSession();
            if (Input.GetKeyDown(KeyCode.H))
            {
                var show = !controlBar.activeSelf;
                controlBar.SetActive(show);
                eventPanel.SetActive(show);
                if (!show)                       // summoned panels never auto-return
                {
                    personalityPanel.SetActive(false);
                    logPanel.SetActive(false);
                }
            }
        }

        static void TogglePanel(GameObject panel, GameObject other)
        {
            var show = !panel.activeSelf;
            panel.SetActive(show);
            if (show) other.SetActive(false);    // the two share the left side
        }

        void Log(string message)
        {
            logHistory.Add($"[{System.DateTime.Now:HH:mm:ss}] {message}");
            if (logText == null) return;
            var start = Mathf.Max(0, logHistory.Count - LogTailLines);
            logText.text = string.Join("\n",
                logHistory.GetRange(start, logHistory.Count - start));
        }

        void AddPanelHeader(string text)
        {
            var go = DefaultControls.CreateText(new DefaultControls.Resources());
            var t = go.GetComponent<Text>();
            t.text = text;
            t.color = Color.white;
            t.fontStyle = FontStyle.Bold;
            t.alignment = TextAnchor.MiddleLeft;
            AttachPanelRow(go, 26f);
        }

        void AddPanelButton(string label, UnityAction onClick)
        {
            var go = DefaultControls.CreateButton(new DefaultControls.Resources());
            go.GetComponentInChildren<Text>().text = label;
            go.GetComponent<Button>().onClick.AddListener(onClick);
            AttachPanelRow(go, 30f);
        }

        void AttachPanelRow(GameObject go, float height)
        {
            go.transform.SetParent(eventPanelContent, false);
            var rt = go.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(rt.sizeDelta.x, height);
        }

        void SetStatus(string msg)
        {
            if (statusText != null) statusText.text = msg;
        }
    }
}
