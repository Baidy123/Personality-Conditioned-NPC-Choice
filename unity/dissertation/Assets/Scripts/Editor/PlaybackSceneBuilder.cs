using System.IO;
using Dissertation.Playback;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Dissertation.EditorTools
{
    // One-click placeholder scene: camera, ground, location blocks + name
    // plates from LocationLayout, NPC, control-bar UI, wired controller.
    // Rerunnable — layout tweaks in LocationLayout.cs, then run again.
    public static class PlaybackSceneBuilder
    {
        const string SpriteDir = "Assets/Sprites";
        const string ScenePath = "Assets/Scenes/Playback.unity";

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
            const float npcScale = 0.55f;
            var agents = new NpcAgent[slotColors.Length];
            var badges = new GameObject[slotColors.Length];
            for (var i = 0; i < slotColors.Length; i++)
            {
                var slotName = ((char)('A' + i)).ToString();
                var npcGo = NewSprite("NPC_" + slotName, circle, slotColors[i]);
                npcGo.transform.position = LocationLayout.NpcStart;
                npcGo.transform.localScale = Vector3.one * npcScale;
                npcGo.GetComponent<SpriteRenderer>().sortingOrder = 5;
                agents[i] = npcGo.AddComponent<NpcAgent>();

                var badge = NewText("Badge", font, slotName, 0.08f);
                badge.transform.SetParent(npcGo.transform, false);
                badge.transform.localScale = Vector3.one / npcScale;   // undo parent scale
                badge.transform.localPosition = Vector3.zero;
                badge.GetComponent<MeshRenderer>().sortingOrder = 11;
                badges[i] = badge;

                var label = NewText("ActionLabel", font, "", 0.11f);
                label.transform.SetParent(npcGo.transform, false);
                label.transform.localScale = Vector3.one / npcScale;
                // each slot's label lives on its own global height plane so
                // co-located NPCs' action words never overlap; localPosition is
                // in parent units, hence the divide by the parent scale
                var worldHeight = LocationLayout.LabelPlanes[i]
                                  - LocationLayout.SlotOffsets[i].y;
                label.transform.localPosition =
                    new Vector3(0f, worldHeight / npcScale, 0f);
                var labelTm = label.GetComponent<TextMesh>();
                labelTm.color = labelColors[i];
                agents[i].actionLabel = labelTm;
            }

            // ------------------------------------------------------------- UI
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

            var dropdowns = new Dropdown[agents.Length];
            for (var i = 0; i < agents.Length; i++)
            {
                var x = 20f + i * 265f;
                var capGo = DefaultControls.CreateText(res);
                var cap = capGo.GetComponent<Text>();
                cap.text = ((char)('A' + i)).ToString();
                cap.color = Color.white;
                cap.alignment = TextAnchor.MiddleCenter;
                Place(capGo, bar.transform, x, 25f);
                var dropdownGo = DefaultControls.CreateDropdown(res);
                Place(dropdownGo, bar.transform, x + 30f, 215f);
                dropdowns[i] = dropdownGo.GetComponent<Dropdown>();
            }

            var continueGo = DefaultControls.CreateButton(res);
            continueGo.GetComponentInChildren<Text>().text = "Continue (Space)";
            Place(continueGo, bar.transform, 830f, 200f);

            var restartGo = DefaultControls.CreateButton(res);
            restartGo.GetComponentInChildren<Text>().text = "Restart (R)";
            Place(restartGo, bar.transform, 1050f, 150f);

            var statusGo = DefaultControls.CreateText(res);
            var status = statusGo.GetComponent<Text>();
            status.text = "";
            status.color = Color.white;
            status.alignment = TextAnchor.MiddleLeft;
            Place(statusGo, bar.transform, 1230f, 670f);

            var ctrl = new GameObject("PlaybackController")
                .AddComponent<PlaybackController>();
            ctrl.npcs = agents;
            ctrl.dropdowns = dropdowns;
            ctrl.continueButton = continueGo.GetComponent<Button>();
            ctrl.restartButton = restartGo.GetComponent<Button>();
            ctrl.statusText = status;
            ctrl.controlBar = bar;
            ctrl.badges = badges;

            Directory.CreateDirectory("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            Debug.Log($"Playback scene built and saved to {ScenePath}");
        }

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
