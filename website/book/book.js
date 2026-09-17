
(function () {
  var RE = new RegExp(
    '(\\/\\/[^\\n]*|#[^\\n]*)' +
    '|("(?:[^"\\\\]|\\\\.)*")' +
    '|(\\b(?:fn|let|mut|while|for|perform|effect|handle|resume|match|from|import|module|export|try|catch|if|else|in|with|performs|struct|enum|trait|implement|implements|return|unsafe|extern|exclave|where|type)\\b)' +
    '|(@[a-z_]+)' +
    '|\\b(\\d+(?:\\.\\d+)?)\\b' +
    '|\\b([A-Z][A-Za-z0-9_]*)\\b',
    'g');
  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  document.querySelectorAll('code.mx').forEach(function (el) {
    var src = el.textContent, out = '', last = 0, m;
    RE.lastIndex = 0;
    while ((m = RE.exec(src)) !== null) {
      out += esc(src.slice(last, m.index));
      var cls = m[1] ? 'tok-c' : m[2] ? 'tok-s' : m[3] ? 'tok-k'
              : m[4] ? 'tok-m' : m[5] ? 'tok-n' : 'tok-t';
      out += '<span class="' + cls + '">' + esc(m[0]) + '</span>';
      last = m.index + m[0].length;
    }
    out += esc(src.slice(last));
    el.innerHTML = out;
  });
})();
