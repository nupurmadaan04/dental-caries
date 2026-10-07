import React, { useMemo } from 'react';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

/**
 * Parses inline markdown tokens (bold, italic, inline code) into React elements.
 */
function renderInline(text: string): React.ReactNode[] {
  // Regex tokenizes inline code (`code`), bold-italic (***text***), bold (**text**), and italic (*text*)
  const tokens: React.ReactNode[] = [];
  let remaining = text;
  let keyIndex = 0;

  while (remaining.length > 0) {
    // 1. Inline code: `...`
    const codeMatch = remaining.match(/^`([^`]+)`/);
    if (codeMatch) {
      tokens.push(
        <code
          key={`code-${keyIndex++}`}
          className="px-1.5 py-0.5 rounded bg-slate-200/80 dark:bg-slate-800 text-cyan-700 dark:text-cyan-300 font-mono text-[11px]"
        >
          {codeMatch[1]}
        </code>
      );
      remaining = remaining.slice(codeMatch[0].length);
      continue;
    }

    // 2. Bold-Italic: ***...*** or ___...___
    const boldItalicMatch = remaining.match(/^(\*\*\*|___)(.+?)\1/);
    if (boldItalicMatch) {
      tokens.push(
        <strong key={`bi-${keyIndex++}`} className="font-bold italic text-slate-900 dark:text-white">
          {renderInline(boldItalicMatch[2])}
        </strong>
      );
      remaining = remaining.slice(boldItalicMatch[0].length);
      continue;
    }

    // 3. Bold: **...** or __...__
    const boldMatch = remaining.match(/^(\*\*|__)(.+?)\1/);
    if (boldMatch) {
      tokens.push(
        <strong key={`b-${keyIndex++}`} className="font-semibold text-slate-900 dark:text-white">
          {renderInline(boldMatch[2])}
        </strong>
      );
      remaining = remaining.slice(boldMatch[0].length);
      continue;
    }

    // 4. Italic: *...* or _..._
    const italicMatch = remaining.match(/^(\*|_)([^\s*_].*?|[^\s*_])\1(?!\*|_)/);
    if (italicMatch) {
      tokens.push(
        <em key={`i-${keyIndex++}`} className="italic text-slate-800 dark:text-slate-200">
          {renderInline(italicMatch[2])}
        </em>
      );
      remaining = remaining.slice(italicMatch[0].length);
      continue;
    }

    // Normal text slice up to next possible markdown character
    const nextSpecial = remaining.search(/[`*_]/);
    if (nextSpecial === -1) {
      tokens.push(remaining);
      break;
    } else if (nextSpecial === 0) {
      // Standalone unmatched markdown character (e.g. streaming or isolated asterisk)
      tokens.push(remaining[0]);
      remaining = remaining.slice(1);
    } else {
      tokens.push(remaining.slice(0, nextSpecial));
      remaining = remaining.slice(nextSpecial);
    }
  }

  return tokens;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  const blocks = useMemo(() => {
    if (!content) return null;

    const lines = content.split('\n');
    const elements: React.ReactNode[] = [];
    let i = 0;
    let blockIndex = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Fenced Code Block
      if (line.trim().startsWith('```')) {
        const lang = line.trim().slice(3).trim();
        const codeLines: string[] = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }
        i++; // consume closing ```
        elements.push(
          <div
            key={`code-block-${blockIndex++}`}
            className="my-2 p-3 rounded-xl bg-slate-900 border border-slate-800 text-slate-100 font-mono text-[11px] overflow-x-auto"
          >
            {lang && (
              <div className="text-[10px] text-slate-400 font-sans uppercase mb-1">{lang}</div>
            )}
            <pre className="whitespace-pre">{codeLines.join('\n')}</pre>
          </div>
        );
        continue;
      }

      // Headings
      if (line.startsWith('### ')) {
        elements.push(
          <h3
            key={`h3-${blockIndex++}`}
            className="text-xs font-bold text-slate-900 dark:text-white mt-2.5 mb-1"
          >
            {renderInline(line.slice(4))}
          </h3>
        );
        i++;
        continue;
      }
      if (line.startsWith('## ')) {
        elements.push(
          <h2
            key={`h2-${blockIndex++}`}
            className="text-sm font-bold text-slate-900 dark:text-white mt-3 mb-1"
          >
            {renderInline(line.slice(3))}
          </h2>
        );
        i++;
        continue;
      }
      if (line.startsWith('# ')) {
        elements.push(
          <h1
            key={`h1-${blockIndex++}`}
            className="text-base font-extrabold text-slate-900 dark:text-white mt-3 mb-1.5"
          >
            {renderInline(line.slice(2))}
          </h1>
        );
        i++;
        continue;
      }

      // Bullet List item (* , - , + , • )
      const bulletMatch = line.match(/^(\s*)([*+-]|•)\s+(.*)$/);
      if (bulletMatch) {
        const listItems: React.ReactNode[] = [];
        while (i < lines.length) {
          const curMatch = lines[i].match(/^(\s*)([*+-]|•)\s+(.*)$/);
          if (!curMatch) break;
          listItems.push(
            <li key={`li-${listItems.length}`} className="flex items-start gap-2 leading-relaxed">
              <span className="text-cyan-500 font-bold select-none mt-0.5">•</span>
              <span className="flex-1">{renderInline(curMatch[3])}</span>
            </li>
          );
          i++;
        }
        elements.push(
          <ul key={`ul-${blockIndex++}`} className="space-y-1 my-1.5 pl-1">
            {listItems}
          </ul>
        );
        continue;
      }

      // Numbered list item (1. )
      const numMatch = line.match(/^(\s*)(\d+)\.\s+(.*)$/);
      if (numMatch) {
        const listItems: React.ReactNode[] = [];
        while (i < lines.length) {
          const curMatch = lines[i].match(/^(\s*)(\d+)\.\s+(.*)$/);
          if (!curMatch) break;
          listItems.push(
            <li key={`ol-${listItems.length}`} className="flex items-start gap-2 leading-relaxed">
              <span className="text-cyan-600 dark:text-cyan-400 font-semibold font-mono text-[11px] min-w-[1.25rem] select-none">
                {curMatch[2]}.
              </span>
              <span className="flex-1">{renderInline(curMatch[3])}</span>
            </li>
          );
          i++;
        }
        elements.push(
          <ol key={`ol-block-${blockIndex++}`} className="space-y-1 my-1.5 pl-1">
            {listItems}
          </ol>
        );
        continue;
      }

      // Empty line / paragraph separator
      if (line.trim() === '') {
        i++;
        continue;
      }

      // Regular Paragraph or text line
      const paraLines: string[] = [line];
      i++;
      while (
        i < lines.length &&
        lines[i].trim() !== '' &&
        !lines[i].trim().startsWith('```') &&
        !lines[i].startsWith('#') &&
        !lines[i].match(/^(\s*)([*+-]|•)\s+/) &&
        !lines[i].match(/^(\s*)(\d+)\.\s+/)
      ) {
        paraLines.push(lines[i]);
        i++;
      }

      elements.push(
        <p key={`p-${blockIndex++}`} className="my-1.5 leading-relaxed">
          {paraLines.map((pLine, pIdx) => (
            <React.Fragment key={`pl-${pIdx}`}>
              {renderInline(pLine)}
              {pIdx < paraLines.length - 1 && <br />}
            </React.Fragment>
          ))}
        </p>
      );
    }

    return elements;
  }, [content]);

  return <div className={`markdown-content text-xs ${className}`}>{blocks}</div>;
};

export default MarkdownRenderer;
