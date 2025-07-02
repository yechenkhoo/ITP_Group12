document.addEventListener('DOMContentLoaded', () => {
    const comparePage = document.getElementById('compare-page');
    if (!comparePage) return console.error('No #compare-page');

    // 1) Parse JSON from data-attrs
    const video1Data = JSON.parse(comparePage.dataset.video1 || '[]');
    const video2Data = JSON.parse(comparePage.dataset.video2 || '[]');

    // 2) Grab URLs & titles
    const video1Url = comparePage.dataset.video1Url;
    const video2Url = comparePage.dataset.video2Url;
    const video1Title = comparePage.dataset.video1Title;
    const video2Title = comparePage.dataset.video2Title;

    // 3) Get video player elements
    const vid1 = document.getElementById('video-player-1');
    const vid2 = document.getElementById('video-player-2');
    if (!vid1 || !vid2) return console.error('Missing <video> tags');

    vid1.src = video1Url;
    vid2.src = video2Url;

    let seeking = false;
    let chartSeeking = false; // New flag to prevent sync for chart clicks

    function syncTime(sourceVideo, targetVideo) {
        if (!seeking && !chartSeeking) { // Add chartSeeking to the condition
            seeking = true;
            targetVideo.currentTime = sourceVideo.currentTime;
            setTimeout(() => seeking = false, 100);
        }
    }

    // Keep the seeking synchronization for manual scrubbing
    ['seeking', 'seeked'].forEach(evt => {
        vid1.addEventListener(evt, () => syncTime(vid1, vid2));
        vid2.addEventListener(evt, () => syncTime(vid2, vid1));
    });

    // 4) Table helpers (No changes needed here)
    function makeCell(val, cls = []) {
        const td = document.createElement('td');
        td.textContent = (typeof val === 'number') ? val.toFixed(2)
            : (val == null || val === '') ? '-' : val;
        td.classList.add(...cls, 'px-6', 'py-4', 'whitespace-nowrap', 'text-sm', 'text-gray-700');
        return td;
    }
    function makeBadge(status) {
        const div = document.createElement('div');
        div.classList.add('inline-flex', 'items-center', 'px-3', 'py-1', 'text-xs', 'font-medium', 'rounded-full');
        if (status === 'Good') div.classList.add('bg-green-100', 'text-green-800');
        else if (status === 'Neutral') div.classList.add('bg-gray-200', 'text-gray-800');
        else if (status === 'Bad') div.classList.add('bg-red-100', 'text-red-800');
        else div.classList.add('bg-gray-200', 'text-gray-800'), status = '-';
        div.textContent = status;
        const td = document.createElement('td');
        td.classList.add('px-6', 'py-4', 'whitespace-nowrap');
        td.appendChild(div);
        return td;
    }
    function getStatus(diff) {
        if (diff > 0) return 'Good';
        if (diff < 0) return 'Bad';
        if (diff === 0) return 'Neutral';
        return '-';
    }

    // 5) Build combined table
    const tbody = document.getElementById('combined-data-body');
    const poseClasses = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'];
    const tiltTypes = ['Shoulder Tilt', 'Hip Tilt'];
    const map1 = new Map(video1Data.map(r => [r['Pose Class'], r]));
    const map2 = new Map(video2Data.map(r => [r['Pose Class'], r]));

    tbody.innerHTML = '';
    if (!video1Data.length && !video2Data.length) {
        const tr = document.createElement('tr');
        const td = makeCell('No data to compare.', ['text-center']);
        td.colSpan = 6;
        tr.appendChild(td);
        tbody.appendChild(tr);
    } else {
        poseClasses.forEach(pose => {
            tiltTypes.forEach(metric => {
                const tr = document.createElement('tr');
                tr.classList.add('border-b', 'hover:bg-indigo-200', 'cursor-pointer');

                const timestamp1 = map1.has(pose) ? map1.get(pose)['Time Frame'] : null;
                const timestamp2 = map2.has(pose) ? map2.get(pose)['Time Frame'] : null;

                // Add data attributes for timestamps
                if (timestamp1 !== null) {
                    tr.dataset.timestampV1 = timestamp1;
                }
                if (timestamp2 !== null) {
                    tr.dataset.timestampV2 = timestamp2;
                }

                tr.appendChild(makeCell(pose, ['font-medium', 'text-gray-900']));
                tr.appendChild(makeCell(metric, ['font-medium', 'text-gray-900']));

                const v1 = parseFloat(map1.get(pose)?.[metric]);
                const v2 = parseFloat(map2.get(pose)?.[metric]);
                tr.appendChild(makeCell(isNaN(v1) ? null : v1));
                tr.appendChild(makeCell(isNaN(v2) ? null : v2));

                let diff = null;
                if (!isNaN(v1) && !isNaN(v2)) diff = +(v2 - v1).toFixed(2);
                tr.appendChild(makeCell(diff));

                const status = (diff == null) ? '-' : getStatus(diff);
                tr.appendChild(makeBadge(status));

                if (status === 'Good') tr.classList.add('bg-green-100');
                else if (status === 'Bad') tr.classList.add('bg-red-100');

                // Add click event listener to the row
                tr.addEventListener('click', () => {
                    const ts1 = parseFloat(tr.dataset.timestampV1);
                    const ts2 = parseFloat(tr.dataset.timestampV2);

                    // For table rows, we still want to seek both videos if data exists
                    if (!isNaN(ts1) && vid1) {
                        vid1.currentTime = ts1;
                        vid1.pause();
                    }
                    if (!isNaN(ts2) && vid2) {
                        vid2.currentTime = ts2;
                        vid2.pause();
                    }
                });

                tbody.appendChild(tr);
            });
        });
    }

    // 6) Tab switching (No changes needed here)
    document.getElementById('table-tab').addEventListener('click', () => {
        document.getElementById('table-view').classList.remove('hidden');
        document.getElementById('chart-view').classList.add('hidden');
        document.getElementById('table-tab').classList.replace('text-gray-500', 'text-blue-600');
        document.getElementById('table-tab').classList.replace('border-transparent', 'border-blue-600');
        document.getElementById('chart-tab').classList.replace('text-blue-600', 'text-gray-500');
        document.getElementById('chart-tab').classList.replace('border-blue-600', 'border-transparent');
    });
    document.getElementById('chart-tab').addEventListener('click', () => {
        document.getElementById('chart-view').classList.remove('hidden');
        document.getElementById('table-view').classList.add('hidden');
        document.getElementById('chart-tab').classList.replace('text-gray-500', 'text-blue-600');
        document.getElementById('chart-tab').classList.replace('border-transparent', 'border-blue-600');
        document.getElementById('table-tab').classList.replace('text-blue-600', 'text-gray-500');
        document.getElementById('table-tab').classList.replace('border-blue-600', 'border-transparent');
    });

    // 7) Build Line Chart for Shoulder and Hip Tilt
    const video1HipTilt = poseClasses.map(pose => parseFloat(map1.get(pose)?.['Hip Tilt']));
    const video2HipTilt = poseClasses.map(pose => parseFloat(map2.get(pose)?.['Hip Tilt']));
    const video1ShoulderTilt = poseClasses.map(pose => parseFloat(map1.get(pose)?.['Shoulder Tilt']));
    const video2ShoulderTilt = poseClasses.map(pose => parseFloat(map2.get(pose)?.['Shoulder Tilt']));

    const ctx = document.getElementById('tiltChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: poseClasses,
            datasets: [
                {
                    label: `${video1Title} - Shoulder Tilt`,
                    data: video1ShoulderTilt,
                    fill: false,
                    borderColor: 'rgb(34, 197, 94)', // Green for Video 1 Shoulder
                    backgroundColor: 'rgb(34, 197, 94)',
                    tension: 0.2
                },
                {
                    label: `${video2Title} - Shoulder Tilt`,
                    data: video2ShoulderTilt,
                    fill: false,
                    borderColor: 'rgb(251, 191, 36)', // Yellow for Video 2 Shoulder
                    backgroundColor: 'rgb(251, 191, 36)',
                    tension: 0.2
                },
                {
                    label: `${video1Title} - Hip Tilt`,
                    data: video1HipTilt,
                    fill: false,
                    borderColor: 'rgb(59, 130, 246)', // Blue for Video 1 Hip
                    backgroundColor: 'rgb(59, 130, 246)',
                    tension: 0.2
                },
                {
                    label: `${video2Title} - Hip Tilt`,
                    data: video2HipTilt,
                    fill: false,
                    borderColor: 'rgb(239, 68, 68)', // Red for Video 2 Hip
                    backgroundColor: 'rgb(239, 68, 68)',
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Tilt Angle'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Pose Class'
                    }
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    chartSeeking = true; // Set flag to disable general sync
                    const firstElement = elements[0];
                    const datasetIndex = firstElement.datasetIndex;
                    const index = firstElement.index;
                    const poseClass = poseClasses[index];

                    if (datasetIndex === 0 || datasetIndex === 2) { // Video 1 datasets
                        const ts1 = map1.has(poseClass) ? parseFloat(map1.get(poseClass)['Time Frame']) : null;
                        if (!isNaN(ts1) && vid1) {
                            vid1.currentTime = ts1;
                            vid1.pause();
                        }
                    } else if (datasetIndex === 1 || datasetIndex === 3) { // Video 2 datasets
                        const ts2 = map2.has(poseClass) ? parseFloat(map2.get(poseClass)['Time Frame']) : null;
                        if (!isNaN(ts2) && vid2) {
                            vid2.currentTime = ts2;
                            vid2.pause();
                        }
                    }
                    // Reset the flag after a short delay to allow sync to re-engage for manual scrubbing
                    setTimeout(() => chartSeeking = false, 200); // Give it a bit more time than 'seeking'
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });

    // Show table by default
    document.getElementById('table-tab').click();
});