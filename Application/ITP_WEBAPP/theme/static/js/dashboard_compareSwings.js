document.addEventListener('DOMContentLoaded', () => {
  const comparePage = document.getElementById('compare-page');
  if (!comparePage) return console.error('No #compare-page');

  // 1) Parse JSON from data-attrs
  const video1Data = JSON.parse(comparePage.dataset.video1 || '[]');
  const video2Data = JSON.parse(comparePage.dataset.video2 || '[]');

  // 2) Grab URLs & titles
  const video1Url   = comparePage.dataset.video1Url;
  const video2Url   = comparePage.dataset.video2Url;
  const video1Title = comparePage.dataset.video1Title;
  const video2Title = comparePage.dataset.video2Title;

  // 3) Sync players
  const vid1 = document.getElementById('video-player-1');
  const vid2 = document.getElementById('video-player-2');
  if (!vid1 || !vid2) return console.error('Missing <video> tags');

  vid1.src = video1Url;
  vid2.src = video2Url;
  vid1.addEventListener('play',  () => vid2.play());
  vid1.addEventListener('pause', () => vid2.pause());
  vid2.addEventListener('play',  () => vid1.play());
  vid2.addEventListener('pause', () => vid1.pause());
  let seeking = false;
  function sync(a, b) {
    if (!seeking) {
      seeking = true;
      b.currentTime = a.currentTime;
      setTimeout(() => seeking = false, 100);
    }
  }
  ['seeking','seeked'].forEach(evt => {
    vid1.addEventListener(evt, () => sync(vid1, vid2));
    vid2.addEventListener(evt, () => sync(vid2, vid1));
  });

  // 4) Table helpers
  function makeCell(val, cls = []) {
    const td = document.createElement('td');
    td.textContent = (typeof val === 'number') ? val.toFixed(2)
      : (val == null || val === '') ? '-' : val;
    td.classList.add(...cls, 'px-6','py-4','whitespace-nowrap','text-sm','text-gray-700');
    return td;
  }
  function makeBadge(status) {
    const div = document.createElement('div');
    div.classList.add('inline-flex','items-center','px-3','py-1','text-xs','font-medium','rounded-full');
    if (status === 'Good')        div.classList.add('bg-green-100','text-green-800');
    else if (status === 'Neutral') div.classList.add('bg-gray-200','text-gray-800');
    else if (status === 'Bad')     div.classList.add('bg-red-100','text-red-800');
    else                           div.classList.add('bg-gray-200','text-gray-800'), status = '-';
    div.textContent = status;
    const td = document.createElement('td');
    td.classList.add('px-6','py-4','whitespace-nowrap');
    td.appendChild(div);
    return td;
  }
  function getStatus(diff) {
    if (diff >  0) return 'Good';
    if (diff <  0) return 'Bad';
    if (diff === 0) return 'Neutral';
    return '-';
  }

  // 5) Build combined table
  const tbody       = document.getElementById('combined-data-body');
  const poseClasses = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10'];
  const tiltTypes   = ['Shoulder Tilt','Hip Tilt','Time Frame'];
  const map1        = new Map(video1Data.map(r => [r['Pose Class'], r]));
  const map2        = new Map(video2Data.map(r => [r['Pose Class'], r]));

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
        tr.classList.add('border-b','hover:bg-gray-50');

        tr.appendChild(makeCell(pose,   ['font-medium','text-gray-900']));
        tr.appendChild(makeCell(metric, ['font-medium','text-gray-900']));

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

        tbody.appendChild(tr);
      });
    });
  }

  // 6) Tab switching
  document.getElementById('table-tab').addEventListener('click', () => {
    document.getElementById('table-view').classList.remove('hidden');
    document.getElementById('chart-view').classList.add('hidden');
    document.getElementById('table-tab').classList.replace('text-gray-500','text-blue-600');
    document.getElementById('table-tab').classList.replace('border-transparent','border-blue-600');
    document.getElementById('chart-tab').classList.replace('text-blue-600','text-gray-500');
    document.getElementById('chart-tab').classList.replace('border-blue-600','border-transparent');
  });
  document.getElementById('chart-tab').addEventListener('click', () => {
    document.getElementById('chart-view').classList.remove('hidden');
    document.getElementById('table-view').classList.add('hidden');
    document.getElementById('chart-tab').classList.replace('text-gray-500','text-blue-600');
    document.getElementById('chart-tab').classList.replace('border-transparent','border-blue-600');
    document.getElementById('table-tab').classList.replace('text-blue-600','text-gray-500');
    document.getElementById('table-tab').classList.replace('border-blue-600','border-transparent');
  });

  // 7) Build Line Chart of Hip Tilt differences
  const hipTiltDiffs = poseClasses.map(pose => {
    const v1 = parseFloat(map1.get(pose)?.['Hip Tilt']);
    const v2 = parseFloat(map2.get(pose)?.['Hip Tilt']);
    return (!isNaN(v1) && !isNaN(v2)) ? +(v2 - v1).toFixed(2) : null;
  });
  const ctx = document.getElementById('tiltChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: poseClasses,
      datasets: [{
        label: 'Hip Tilt Difference',
        data: hipTiltDiffs,
        fill: false,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgb(59, 130, 246)',
        tension: 0.2
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, title: { display: true, text: 'Δ Hip Tilt' } },
        x: { title: { display: true, text: 'Pose Class' } }
      }
    }
  });

  // Show table by default
  document.getElementById('table-tab').click();
});
