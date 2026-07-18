using System.IO;
using Dissertation.Live;
using Dissertation.Playback;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Dissertation.EditorTools
{
    // One-click placeholder scene for BOTH modes: camera, ground, location
    // blocks + name plates from LocationLayout, playback NPCs (A/B/C), the
    // live-NPC prototype, the control bar (mode dropdown + per-mode widget
    // groups), the live event/personality panels, and all component wiring.
    // Rerunnable — tweak LocationLayout.cs or this file, then run again.
    public static class PlaybackSceneBuilder
    {
        const string SpriteDir = "Assets/Sprites";
        const string ScenePath = "Assets/Scenes/Playback.unity";
        const float NpcScale = 0.55f;

        [MenuItem("Dissertation/Build Playback Scene")]
        public static void Build()
        {
            // rebuilding replaces the open scene — don't silently drop edits
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            var square = EnsureSprite("white_square.png", MakeSquareTex());
            var circle = EnsureSprite("circle.png", MakeCircleTex());
            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            var scene = EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            var cam = camGo.AddComponent<Camera>();
            cam.orthographic = true;
            cam.orthographicSize = 5f;
            cam.transform.position = new Vector3(0f, 0f, -10f);
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.13f, 0.16f, 0.13f);

            var ground = NewSprite("Ground", square, new Color(0.21f, 0.26f, 0.21f));
            ground.transform.localScale = new Vector3(19f, 11f, 1f);
            ground.GetComponent<SpriteRenderer>().sortingOrder = -10;

            foreach (var kv in LocationLayout.Entries)
            {
                var block = NewSprite(kv.Key, square, kv.Value.Color);
                block.transform.position = kv.Value.Position;
                block.transform.localScale = new Vector3(2.4f, 1.7f, 1f);
                // plate is not parented: the block's non-uniform scale would distort it
                var plate = NewText(kv.Key + "_label", font, kv.Key, 0.12f);
                plate.transform.position =
                    kv.Value.Position + new Vector2(0f, 1.15f);
            }

            // ------------------------------------------------- playback NPCs --
            var slotColors = new[]
            {
                new Color(0.13f, 0.13f, 0.16f),   // A near-black
                new Color(0.30f, 0.12f, 0.35f),   // B dark purple
                new Color(0.05f, 0.28f, 0.32f),   // C dark teal
            };
            var labelColors = new[]               // light tints of the slot colours
            {
                new Color(0.95f, 0.95f, 0.95f),
                new Color(0.85f, 0.65f, 0.95f),
                new Color(0.55f, 0.95f, 0.90f),
            };
            var agents = new NpcAgent[slotColors.Length];
            var badges = new GameObject[slotColors.Length];
            for (var i = 0; i < slotColors.Length; i++)
            {
                var slotName = ((char)('A' + i)).ToString();
                var npcGo = NewSprite("NPC_" + slotName, circle, slotColors[i]);
                npcGo.transform.position = LocationLayout.NpcStart;
                npcGo.transform.localScale = Vector3.one * NpcScale;
                npcGo.GetComponent<SpriteRenderer>().sortingOrder = 5;
                agents[i] = npcGo.AddComponent<NpcAgent>();

                var badge = NewText("Badge", font, slotName, 0.08f);
                badge.transform.SetParent(npcGo.transform, false);
                badge.transform.localScale = Vector3.one / NpcScale;   // undo parent scale
                badge.transform.localPosition = Vector3.zero;
                badge.GetComponent<MeshRenderer>().sortingOrder = 11;
                badges[i] = badge;

                // each slot's label lives on its own global height plane so
                // co-located NPCs' action words never overlap; localPosition is
                // in parent units, hence the divide by the parent scale
                var label = NewText("ActionLabel", font, "", 0.11f);
                label.transform.SetParent(npcGo.transform, false);
                label.transform.localScale = Vector3.one / NpcScale;
                var worldHeight = LocationLayout.LabelPlanes[i]
                                  - LocationLayout.SlotOffsets[i].y;
                label.transform.localPosition =
                    new Vector3(0f, worldHeight / NpcScale, 0f);
                var labelTm = label.GetComponent<TextMesh>();
                labelTm.color = labelColors[i];
                agents[i].actionLabel = labelTm;
            }

            // -------------------------------------------- live NPC prototype --
            // LiveController instantiates one copy per NPC the server reports.
            var proto = NewSprite("LiveNpcPrototype", circle, Color.white);
            proto.transform.position = LocationLayout.NpcStart;
            proto.transform.localScale = Vector3.one * NpcScale;
            proto.GetComponent<SpriteRenderer>().sortingOrder = 5;
            var protoAgent = proto.AddComponent<NpcAgent>();

            var protoAction = NewText("ActionLabel", font, "", 0.11f);
            protoAction.transform.SetParent(proto.transform, false);
            protoAction.transform.localScale = Vector3.one / NpcScale;
            protoAction.transform.localPosition =
                new Vector3(0f, LocationLayout.LabelPlanes[0] / NpcScale, 0f);
            protoAgent.actionLabel = protoAction.GetComponent<TextMesh>();

            var protoName = NewText("NameLabel", font, "", 0.09f);
            protoName.transform.SetParent(proto.transform, false);
            protoName.transform.localScale = Vector3.one / NpcScale;
            protoName.transform.localPosition = new Vector3(0f, -0.6f / NpcScale, 0f);
            proto.SetActive(false);

            // ------------------------------------------------------------- UI --
            var canvasGo = new GameObject("Canvas",
                typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasGo.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGo.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920f, 1080f);

            new GameObject("EventSystem",
                typeof(EventSystem), typeof(StandaloneInputModule));

            var bar = new GameObject("ControlBar", typeof(RectTransform), typeof(Image));
            bar.transform.SetParent(canvasGo.transform, false);
            var barRt = bar.GetComponent<RectTransform>();
            barRt.anchorMin = new Vector2(0f, 0f);
            barRt.anchorMax = new Vector2(1f, 0f);
            barRt.pivot = new Vector2(0.5f, 0f);
            barRt.sizeDelta = new Vector2(0f, 90f);
            barRt.anchoredPosition = Vector2.zero;
            bar.GetComponent<Image>().color = new Color(0f, 0f, 0f, 0.6f);

            var res = new DefaultControls.Resources();

            // shared: mode dropdown (far left) + status line (far right)
            var modeGo = DefaultControls.CreateDropdown(res);
            Place(modeGo, bar.transform, 20f, 150f);

            var statusGo = DefaultControls.CreateText(res);
            var status = statusGo.GetComponent<Text>();
            status.text = "";
            status.color = Color.white;
            status.alignment = TextAnchor.MiddleLeft;
            Place(statusGo, bar.transform, 1470f, 440f);

            // ------------------------------------------- playback bar widgets --
            var playbackGroup = Group("PlaybackGroup", bar.transform);

            var countCapGo = DefaultControls.CreateText(res);
            var countCap = countCapGo.GetComponent<Text>();
            countCap.text = "NPCs";
            countCap.color = Color.white;
            countCap.alignment = TextAnchor.MiddleCenter;
            Place(countCapGo, playbackGroup.transform, 185f, 60f);

            var countGo = DefaultControls.CreateDropdown(res);
            Place(countGo, playbackGroup.transform, 250f, 85f);

            var dropdowns = new Dropdown[agents.Length];
            var slotRows = new GameObject[agents.Length];
            for (var i = 0; i < agents.Length; i++)
            {
                var row = new GameObject("SlotRow_" + (char)('A' + i),
                    typeof(RectTransform));
                Place(row, playbackGroup.transform, 360f + i * 260f, 250f);
                slotRows[i] = row;

                var capGo = DefaultControls.CreateText(res);
                var cap = capGo.GetComponent<Text>();
                cap.text = ((char)('A' + i)).ToString();
                cap.color = Color.white;
                cap.alignment = TextAnchor.MiddleCenter;
                Place(capGo, row.transform, 0f, 25f);
                var dropdownGo = DefaultControls.CreateDropdown(res);
                Place(dropdownGo, row.transform, 30f, 215f);
                dropdowns[i] = dropdownGo.GetComponent<Dropdown>();
            }

            var continueGo = DefaultControls.CreateButton(res);
            continueGo.GetComponentInChildren<Text>().text = "Continue (Space)";
            Place(continueGo, playbackGroup.transform, 1150f, 160f);

            var restartGo = DefaultControls.CreateButton(res);
            restartGo.GetComponentInChildren<Text>().text = "Restart (R)";
            Place(restartGo, playbackGroup.transform, 1320f, 130f);

            // ----------------------------------------------- live bar widgets --
            var liveGroup = Group("LiveGroup", bar.transform);

            var connectGo = DefaultControls.CreateButton(res);
            connectGo.GetComponentInChildren<Text>().text = "Connect";
            Place(connectGo, liveGroup.transform, 185f, 140f);

            var stepGo = DefaultControls.CreateButton(res);
            stepGo.GetComponentInChildren<Text>().text = "Step (Space)";
            Place(stepGo, liveGroup.transform, 340f, 160f);

            var resetGo = DefaultControls.CreateButton(res);
            resetGo.GetComponentInChildren<Text>().text = "Reset (R)";
            Place(resetGo, liveGroup.transform, 515f, 130f);

            var persToggleGo = DefaultControls.CreateButton(res);
            persToggleGo.GetComponentInChildren<Text>().text = "Personality";
            Place(persToggleGo, liveGroup.transform, 660f, 140f);

            var logToggleGo = DefaultControls.CreateButton(res);
            logToggleGo.GetComponentInChildren<Text>().text = "Log";
            Place(logToggleGo, liveGroup.transform, 815f, 90f);

            // ------------------------------------------------ event panel (R) --
            var eventPanel = new GameObject("EventPanel", typeof(RectTransform),
                typeof(Image), typeof(VerticalLayoutGroup));
            eventPanel.transform.SetParent(canvasGo.transform, false);
            var epRt = eventPanel.GetComponent<RectTransform>();
            epRt.anchorMin = new Vector2(1f, 0f);
            epRt.anchorMax = new Vector2(1f, 1f);
            epRt.offsetMin = new Vector2(-320f, 100f);
            epRt.offsetMax = new Vector2(-10f, -10f);
            eventPanel.GetComponent<Image>().color = new Color(0f, 0f, 0f, 0.55f);
            var vlg = eventPanel.GetComponent<VerticalLayoutGroup>();
            vlg.padding = new RectOffset(8, 8, 8, 8);
            vlg.spacing = 4f;
            vlg.childControlWidth = true;
            vlg.childForceExpandWidth = true;
            vlg.childControlHeight = false;
            vlg.childForceExpandHeight = false;
            vlg.childAlignment = TextAnchor.UpperLeft;

            // ------------------------------------------ personality panel (L) --
            var persPanel = new GameObject("PersonalityPanel",
                typeof(RectTransform), typeof(Image));
            persPanel.transform.SetParent(canvasGo.transform, false);
            var ppRt = persPanel.GetComponent<RectTransform>();
            ppRt.anchorMin = new Vector2(0f, 0f);
            ppRt.anchorMax = new Vector2(0f, 1f);
            ppRt.offsetMin = new Vector2(10f, 100f);
            ppRt.offsetMax = new Vector2(320f, -10f);
            persPanel.GetComponent<Image>().color = new Color(0f, 0f, 0f, 0.55f);

            var titleGo = DefaultControls.CreateText(res);
            var title = titleGo.GetComponent<Text>();
            title.text = "Personality";
            title.color = Color.white;
            title.fontStyle = FontStyle.Bold;
            title.alignment = TextAnchor.MiddleLeft;
            PlaceAt(titleGo, persPanel.transform, 10f, 10f, 280f, 26f);

            var npcDdGo = DefaultControls.CreateDropdown(res);
            PlaceAt(npcDdGo, persPanel.transform, 10f, 42f, 280f, 32f);

            var traitNames = new[] { "O", "C", "E", "A", "N" };
            var sliders = new Slider[5];
            var valueLabels = new Text[5];
            for (var i = 0; i < 5; i++)
            {
                var y = 86f + i * 38f;
                var capGo = DefaultControls.CreateText(res);
                var cap = capGo.GetComponent<Text>();
                cap.text = traitNames[i];
                cap.color = Color.white;
                cap.alignment = TextAnchor.MiddleCenter;
                PlaceAt(capGo, persPanel.transform, 10f, y, 25f, 30f);

                var sliderGo = DefaultControls.CreateSlider(res);
                PlaceAt(sliderGo, persPanel.transform, 42f, y + 6f, 178f, 20f);
                var slider = sliderGo.GetComponent<Slider>();
                slider.minValue = -1f;
                slider.maxValue = 1f;
                slider.value = 0f;
                Tint(sliderGo, "Background", new Color(0.35f, 0.35f, 0.35f));
                Tint(sliderGo, "Fill Area/Fill", new Color(0.35f, 0.6f, 0.9f));
                sliders[i] = slider;

                var valGo = DefaultControls.CreateText(res);
                var val = valGo.GetComponent<Text>();
                val.text = "+0.00";
                val.color = Color.white;
                val.alignment = TextAnchor.MiddleRight;
                PlaceAt(valGo, persPanel.transform, 228f, y, 62f, 30f);
                valueLabels[i] = val;
            }

            var applyGo = DefaultControls.CreateButton(res);
            applyGo.GetComponentInChildren<Text>().text = "Apply";
            PlaceAt(applyGo, persPanel.transform, 10f, 286f, 135f, 32f);

            var nameInputGo = DefaultControls.CreateInputField(res);
            PlaceAt(nameInputGo, persPanel.transform, 10f, 330f, 180f, 32f);
            var nameInput = nameInputGo.GetComponent<InputField>();
            var placeholder = nameInput.placeholder as Text;
            if (placeholder != null) placeholder.text = "new NPC name...";

            var addGo = DefaultControls.CreateButton(res);
            addGo.GetComponentInChildren<Text>().text = "Add NPC";
            PlaceAt(addGo, persPanel.transform, 200f, 330f, 90f, 32f);

            // ------------------------------------------------- log panel (L) --
            // shares the left side with the personality panel; the controller
            // keeps at most one of the two visible
            var logPanel = new GameObject("LogPanel",
                typeof(RectTransform), typeof(Image));
            logPanel.transform.SetParent(canvasGo.transform, false);
            var lpRt = logPanel.GetComponent<RectTransform>();
            lpRt.anchorMin = new Vector2(0f, 0f);
            lpRt.anchorMax = new Vector2(0f, 1f);
            lpRt.offsetMin = new Vector2(10f, 100f);
            lpRt.offsetMax = new Vector2(480f, -10f);
            logPanel.GetComponent<Image>().color = new Color(0f, 0f, 0f, 0.65f);

            var logTextGo = DefaultControls.CreateText(res);
            logTextGo.transform.SetParent(logPanel.transform, false);
            var ltRt = logTextGo.GetComponent<RectTransform>();
            ltRt.anchorMin = Vector2.zero;
            ltRt.anchorMax = Vector2.one;
            ltRt.offsetMin = new Vector2(10f, 10f);
            ltRt.offsetMax = new Vector2(-10f, -10f);
            var logText = logTextGo.GetComponent<Text>();
            logText.text = "";
            logText.color = new Color(0.8f, 0.95f, 0.8f);
            logText.fontSize = 15;
            logText.alignment = TextAnchor.LowerLeft;
            logText.horizontalOverflow = HorizontalWrapMode.Wrap;
            logText.verticalOverflow = VerticalWrapMode.Overflow;

            // ------------------------------------------------------- wiring --
            var playCtrl = new GameObject("PlaybackController")
                .AddComponent<PlaybackController>();
            playCtrl.npcs = agents;
            playCtrl.dropdowns = dropdowns;
            playCtrl.countDropdown = countGo.GetComponent<Dropdown>();
            playCtrl.slotRows = slotRows;
            playCtrl.continueButton = continueGo.GetComponent<Button>();
            playCtrl.restartButton = restartGo.GetComponent<Button>();
            playCtrl.statusText = status;
            playCtrl.controlBar = bar;
            playCtrl.badges = badges;

            var liveGo = new GameObject("LiveController");
            var client = liveGo.AddComponent<LiveClient>();
            var launcher = liveGo.AddComponent<BrainLauncher>();
            var liveCtrl = liveGo.AddComponent<LiveController>();
            liveCtrl.client = client;
            liveCtrl.launcher = launcher;
            liveCtrl.npcPrototype = protoAgent;
            liveCtrl.connectButton = connectGo.GetComponent<Button>();
            liveCtrl.stepButton = stepGo.GetComponent<Button>();
            liveCtrl.resetButton = resetGo.GetComponent<Button>();
            liveCtrl.statusText = status;
            liveCtrl.controlBar = bar;
            liveCtrl.eventPanel = eventPanel;
            liveCtrl.eventPanelContent = epRt;
            liveCtrl.personalityPanel = persPanel;
            liveCtrl.personalityToggleButton = persToggleGo.GetComponent<Button>();
            liveCtrl.logPanel = logPanel;
            liveCtrl.logText = logText;
            liveCtrl.logToggleButton = logToggleGo.GetComponent<Button>();
            liveCtrl.npcDropdown = npcDdGo.GetComponent<Dropdown>();
            liveCtrl.traitSliders = sliders;
            liveCtrl.traitValueLabels = valueLabels;
            liveCtrl.applyButton = applyGo.GetComponent<Button>();
            liveCtrl.newNpcName = nameInput;
            liveCtrl.addNpcButton = addGo.GetComponent<Button>();

            var switcher = new GameObject("ModeSwitcher").AddComponent<ModeSwitcher>();
            switcher.modeDropdown = modeGo.GetComponent<Dropdown>();
            switcher.playback = playCtrl;
            switcher.live = liveCtrl;
            switcher.playbackGroup = playbackGroup;
            switcher.liveGroup = liveGroup;
            switcher.eventPanel = eventPanel;
            switcher.personalityPanel = persPanel;
            switcher.logPanel = logPanel;

            // playback is the startup mode; ModeSwitcher re-syncs at runtime
            liveGroup.SetActive(false);
            eventPanel.SetActive(false);
            persPanel.SetActive(false);
            logPanel.SetActive(false);

            Directory.CreateDirectory("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            Debug.Log($"Playback + Live scene built and saved to {ScenePath}");
        }

        // ------------------------------------------------------------ helpers --
        static void Place(GameObject go, Transform parent, float x, float width)
        {
            go.transform.SetParent(parent, false);
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0f, 0.5f);
            rt.anchorMax = new Vector2(0f, 0.5f);
            rt.pivot = new Vector2(0f, 0.5f);
            rt.anchoredPosition = new Vector2(x, 0f);
            rt.sizeDelta = new Vector2(width, 48f);
        }

        // top-left anchored placement inside a panel (y grows downward)
        static void PlaceAt(GameObject go, Transform parent,
                            float x, float y, float width, float height)
        {
            go.transform.SetParent(parent, false);
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0f, 1f);
            rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot = new Vector2(0f, 1f);
            rt.anchoredPosition = new Vector2(x, -y);
            rt.sizeDelta = new Vector2(width, height);
        }

        static GameObject Group(string name, Transform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
            return go;
        }

        static void Tint(GameObject root, string childPath, Color color)
        {
            var child = root.transform.Find(childPath);
            if (child == null) return;
            var image = child.GetComponent<Image>();
            if (image != null) image.color = color;
        }

        static GameObject NewSprite(string name, Sprite sprite, Color color)
        {
            var go = new GameObject(name);
            var sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = sprite;
            sr.color = color;
            return go;
        }

        static GameObject NewText(string name, Font font, string text, float characterSize)
        {
            var go = new GameObject(name);
            var tm = go.AddComponent<TextMesh>();
            tm.text = text;
            tm.font = font;
            tm.fontSize = 64;
            tm.characterSize = characterSize;
            tm.anchor = TextAnchor.MiddleCenter;
            tm.alignment = TextAlignment.Center;
            tm.color = Color.white;
            var mr = go.GetComponent<MeshRenderer>();
            mr.material = font.material;
            mr.sortingOrder = 10;
            return go;
        }

        static Sprite EnsureSprite(string file, Texture2D tex)
        {
            // always rewrite + reimport: idempotent, and self-heals a half-done
            // first run or an edited Make*Tex shape
            Directory.CreateDirectory(SpriteDir);
            var path = $"{SpriteDir}/{file}";
            File.WriteAllBytes(path, tex.EncodeToPNG());
            AssetDatabase.ImportAsset(path);
            var imp = (TextureImporter)AssetImporter.GetAtPath(path);
            imp.textureType = TextureImporterType.Sprite;
            imp.spritePixelsPerUnit = tex.width;   // every sprite is 1x1 world units
            imp.SaveAndReimport();
            Object.DestroyImmediate(tex);
            var sprite = AssetDatabase.LoadAssetAtPath<Sprite>(path);
            if (sprite == null)
                throw new System.InvalidOperationException($"sprite import failed: {path}");
            return sprite;
        }

        static Texture2D MakeSquareTex()
        {
            var tex = new Texture2D(4, 4, TextureFormat.RGBA32, false);
            var pixels = new Color[16];
            for (var i = 0; i < pixels.Length; i++) pixels[i] = Color.white;
            tex.SetPixels(pixels);
            tex.Apply();
            return tex;
        }

        static Texture2D MakeCircleTex()
        {
            const int n = 64;
            var tex = new Texture2D(n, n, TextureFormat.RGBA32, false);
            var half = (n - 1) / 2f;
            var r2 = (n / 2f - 1f) * (n / 2f - 1f);
            for (var y = 0; y < n; y++)
                for (var x = 0; x < n; x++)
                {
                    var dx = x - half;
                    var dy = y - half;
                    tex.SetPixel(x, y, dx * dx + dy * dy <= r2 ? Color.white : Color.clear);
                }
            tex.Apply();
            return tex;
        }
    }
}
