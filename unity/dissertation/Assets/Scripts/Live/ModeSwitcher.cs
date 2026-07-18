using System.Collections.Generic;
using Dissertation.Playback;
using UnityEngine;
using UnityEngine.UI;

namespace Dissertation.Live
{
    // The Mode dropdown: Playback (pre-generated sequence files) or Live
    // (the local Python brain). Only one controller is enabled at a time, so
    // the shared hotkeys (Space/R/H) always reach exactly one handler.
    public class ModeSwitcher : MonoBehaviour
    {
        public Dropdown modeDropdown;
        public PlaybackController playback;
        public LiveController live;
        public GameObject playbackGroup;   // playback's control-bar widgets
        public GameObject liveGroup;       // live's control-bar widgets
        public GameObject eventPanel;
        public GameObject personalityPanel;

        void Start()
        {
            modeDropdown.ClearOptions();
            modeDropdown.AddOptions(new List<string> { "Playback", "Live" });
            modeDropdown.onValueChanged.AddListener(_ => Apply());
            Apply();
        }

        void Apply()
        {
            var liveMode = modeDropdown.value == 1;
            playback.enabled = !liveMode;
            live.enabled = liveMode;
            playbackGroup.SetActive(!liveMode);
            liveGroup.SetActive(liveMode);
            eventPanel.SetActive(liveMode);
            personalityPanel.SetActive(liveMode);
            playback.SetNpcsVisible(!liveMode);
            live.SetNpcsVisible(liveMode);
        }
    }
}
