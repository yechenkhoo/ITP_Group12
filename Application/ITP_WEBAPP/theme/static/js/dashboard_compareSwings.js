document.addEventListener('DOMContentLoaded', () => {
  // Pull everything out of the #compare-page dataset
  const comparePage = document.getElementById('compare-page');
  const data1       = JSON.parse(comparePage.dataset.video1);
  const data2       = JSON.parse(comparePage.dataset.video2);
  const video1Title = comparePage.dataset.video1Title;
  const video2Title = comparePage.dataset.video2Title;
  const video1Url   = comparePage.dataset.video1Url;
  const video2Url   = comparePage.dataset.video2Url;

  // Grab your DOM nodes
  const videoPlayer1     = document.getElementById('video-player-1');
  const videoPlayer2     = document.getElementById('video-player-2');
  const videoTitle1      = document.getElementById('video-title-1');
  const videoTitle2      = document.getElementById('video-title-2');
  const combinedDataBody = document.getElementById('combined-data-body');

  // Wire up the video tags
  videoPlayer1.src       = video1Url;
  videoTitle1.textContent = video1Title;
  videoPlayer2.src       = video2Url;
  videoTitle2.textContent = video2Title;

  // Sync playback
  videoPlayer1.addEventListener('play',  () => videoPlayer2.play());
  videoPlayer1.addEventListener('pause', () => videoPlayer2.pause());
  videoPlayer2.addEventListener('play',  () => videoPlayer1.play());
  videoPlayer2.addEventListener('pause', () => videoPlayer1.pause());
  let isSeeking = false;
  function syncSeek(a,b){
    if(!isSeeking){
      isSeeking = true;
      b.currentTime = a.currentTime;
      setTimeout(()=> isSeeking=false, 100);
    }
  }
  ['seeking','seeked'].forEach(evt=>{
    videoPlayer1.addEventListener(evt, ()=> syncSeek(videoPlayer1, videoPlayer2));
    videoPlayer2.addEventListener(evt, ()=> syncSeek(videoPlayer2, videoPlayer1));
  });

  // Build the combined-data table
  const maxRows = Math.max(data1.length, data2.length);
  for(let i=0; i<maxRows; i++){
    const r1 = data1[i]||{};
    const r2 = data2[i]||{};
    const tr = document.createElement('tr');
    tr.classList.add('border-b','hover:bg-gray-50');

    // helper to create a <td>
    function makeCell(text, ...classes){
      const td = document.createElement('td');
      td.textContent = text||'-';
      td.classList.add(...classes);
      return td;
    }

    // Time Frame
    tr.appendChild(makeCell(r1['Time Frame'] || r2['Time Frame'],
      'px-6','py-4','whitespace-nowrap','text-sm','font-medium','text-gray-900'));

    // Overall Status (Video 1)
    const status1 = r1['Overall Status'];
    const td1 = makeCell('', 'px-6','py-4','whitespace-nowrap');
    const badge1 = document.createElement('div');
    badge1.classList.add('inline-flex','items-center','px-3','py-1','md:text-sm','text-xs','font-medium','rounded-full');
    if      (status1==='Good')     badge1.classList.add('bg-green-100','text-green-800');
    else if (status1==='Bad')      badge1.classList.add('bg-yellow-100','text-yellow-800');
    else if (status1==='Very Bad') badge1.classList.add('bg-red-100','text-red-800');
    else                            badge1.classList.add('bg-gray-200','text-gray-800');
    badge1.textContent = status1||'-';
    td1.appendChild(badge1);
    tr.appendChild(td1);

    // Swing Speed (Video 1)
    tr.appendChild(makeCell(r1['Swing Speed'], 'px-6','py-4','whitespace-nowrap','text-sm','text-gray-700'));

    // Overall Status (Video 2)
    const status2 = r2['Overall Status'];
    const td2 = makeCell('', 'px-6','py-4','whitespace-nowrap');
    const badge2 = badge1.cloneNode();
    badge2.textContent = status2||'-';
    badge2.className = badge1.className; // reset classes
    if      (status2==='Good')     badge2.classList.add('bg-green-100','text-green-800');
    else if (status2==='Bad')      badge2.classList.add('bg-yellow-100','text-yellow-800');
    else if (status2==='Very Bad') badge2.classList.add('bg-red-100','text-red-800');
    else                            badge2.classList.add('bg-gray-200','text-gray-800');
    td2.appendChild(badge2);
    tr.appendChild(td2);

    // Swing Speed (Video 2)
    tr.appendChild(makeCell(r2['Swing Speed'], 'px-6','py-4','whitespace-nowrap','text-sm','text-gray-700'));

    combinedDataBody.appendChild(tr);
  }
});
