(function (global) {
  'use strict';

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function sanitizeUrl(raw) {
    const url = String(raw || '').trim();
    if (!url) return '';
    if (/^(https?:|mailto:|#)/i.test(url)) return url;
    return '';
  }

  function renderInline(text) {
    let out = text;
    out = out.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    out = out.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    out = out.replace(/\[([^\]]+)\]\(([^)\n]+)\)/g, (_, label, href) => {
      const safe = sanitizeUrl(href);
      if (!safe) return label;
      return `<a href="${safe}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    });
    return out;
  }

  function isTableRow(line) {
    const trimmed = line.trim();
    return trimmed.startsWith('|') && trimmed.endsWith('|') && trimmed.includes('|');
  }

  function splitTableRow(line) {
    return line
      .trim()
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map((cell) => cell.trim());
  }

  function isTableSeparator(line) {
    return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line.trim());
  }

  function renderTable(lines) {
    const header = splitTableRow(lines[0]);
    const bodyRows = lines.slice(2).map(splitTableRow);
    const headHtml = header.map((cell) => `<th>${renderInline(cell)}</th>`).join('');
    const bodyHtml = bodyRows.map((row) =>
      `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join('')}</tr>`,
    ).join('');
    return `<div class="md-table-wrap"><table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`;
  }

  function renderMarkdownHtml(text) {
    const source = escapeHtml(String(text ?? '')).replace(/\r\n/g, '\n');
    if (!source.trim()) return '';

    const lines = source.split('\n');
    const blocks = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
        blocks.push('<hr />');
        index += 1;
        continue;
      }

      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        const level = heading[1].length;
        blocks.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        index += 1;
        continue;
      }

      if (isTableRow(line) && index + 1 < lines.length && isTableSeparator(lines[index + 1])) {
        const tableLines = [line, lines[index + 1]];
        index += 2;
        while (index < lines.length && isTableRow(lines[index])) {
          tableLines.push(lines[index]);
          index += 1;
        }
        blocks.push(renderTable(tableLines));
        continue;
      }

      if (/^>\s?/.test(line)) {
        const quoteLines = [];
        while (index < lines.length && /^>\s?/.test(lines[index])) {
          quoteLines.push(lines[index].replace(/^>\s?/, ''));
          index += 1;
        }
        blocks.push(`<blockquote><p>${renderInline(quoteLines.join(' '))}</p></blockquote>`);
        continue;
      }

      if (/^```/.test(line.trim())) {
        const codeLines = [];
        index += 1;
        while (index < lines.length && !/^```/.test(lines[index].trim())) {
          codeLines.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        blocks.push(`<pre><code>${codeLines.join('\n')}</code></pre>`);
        continue;
      }

      const unordered = line.match(/^[-*•]\s+(.+)$/);
      const ordered = line.match(/^\d+\.\s+(.+)$/);
      if (unordered || ordered) {
        const listTag = ordered ? 'ol' : 'ul';
        const items = [];
        while (index < lines.length) {
          const bullet = lines[index].match(/^[-*•]\s+(.+)$/);
          const number = lines[index].match(/^\d+\.\s+(.+)$/);
          const match = ordered ? number : bullet;
          if (!match) break;
          items.push(`<li>${renderInline(match[1])}</li>`);
          index += 1;
        }
        blocks.push(`<${listTag}>${items.join('')}</${listTag}>`);
        continue;
      }

      const paragraph = [];
      while (
        index < lines.length
        && lines[index].trim()
        && !/^(#{1,6})\s/.test(lines[index])
        && !/^(-{3,}|\*{3,}|_{3,})$/.test(lines[index].trim())
        && !/^[-*•]\s/.test(lines[index])
        && !/^\d+\.\s/.test(lines[index])
        && !/^>\s?/.test(lines[index])
        && !/^```/.test(lines[index].trim())
        && !(isTableRow(lines[index]) && index + 1 < lines.length && isTableSeparator(lines[index + 1]))
      ) {
        paragraph.push(lines[index].trim());
        index += 1;
      }
      if (paragraph.length) {
        blocks.push(`<p>${renderInline(paragraph.join(' '))}</p>`);
      }
    }

    return blocks.join('\n');
  }

  global.renderMarkdownHtml = renderMarkdownHtml;
  global.escapeHtml = escapeHtml;
})(window);
