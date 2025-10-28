document.addEventListener('DOMContentLoaded', () => {
    // Select both video players
    const vid1 = document.getElementById('video-player'); // Left video
    const vid2 = document.getElementById('video-player-right'); // Right video

    // Only proceed if *both* video players exist on the page
    if (vid1 && vid2) {
        let seeking = false;

        function syncTime(sourceVideo, targetVideo) {
            if (!seeking) {
                seeking = true;
                targetVideo.currentTime = sourceVideo.currentTime;
                // Use a short timeout to prevent event loops
                setTimeout(() => seeking = false, 100);
            }
        }

        // --- Sync Time (Seeking) ---
        // Add listeners for both seeking and seeked events for robustness
        ['seeking', 'seeked'].forEach(evt => {
            vid1.addEventListener(evt, () => syncTime(vid1, vid2));
            vid2.addEventListener(evt, () => syncTime(vid2, vid1));
        });

        // --- Sync State (Play/Pause) ---
        vid1.addEventListener('play', () => {
            if (vid2.paused) vid2.play();
        });
        vid1.addEventListener('pause', () => {
            if (!vid2.paused) vid2.pause();
        });

        vid2.addEventListener('play', () => {
            if (vid1.paused) vid1.play();
        });
        vid2.addEventListener('pause', () => {
            if (!vid1.paused) vid1.pause();
        });
    }
});