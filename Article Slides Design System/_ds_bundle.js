/* @ds-bundle: {"format":4,"namespace":"ArticleSlidesDesignSystem_a47fec","components":[{"name":"References","sourcePath":"components/backmatter/References.jsx"},{"name":"Admonition","sourcePath":"components/blocks/Admonition.jsx"},{"name":"Aside","sourcePath":"components/blocks/Aside.jsx"},{"name":"Proof","sourcePath":"components/blocks/Proof.jsx"},{"name":"Button","sourcePath":"components/core/Button.jsx"},{"name":"Icon","sourcePath":"components/core/Icon.jsx"},{"name":"ICON_NAMES","sourcePath":"components/core/Icon.jsx"},{"name":"Abstract","sourcePath":"components/frontmatter/Abstract.jsx"},{"name":"FrontmatterBlock","sourcePath":"components/frontmatter/FrontmatterBlock.jsx"},{"name":"Blockquote","sourcePath":"components/media/Blockquote.jsx"},{"name":"CodeBlock","sourcePath":"components/media/CodeBlock.jsx"},{"name":"Equation","sourcePath":"components/media/Equation.jsx"},{"name":"Figure","sourcePath":"components/media/Figure.jsx"},{"name":"FooterLink","sourcePath":"components/slide/FooterLink.jsx"},{"name":"Slide","sourcePath":"components/slide/Slide.jsx"},{"name":"SlideHeader","sourcePath":"components/slide/SlideHeader.jsx"}],"sourceHashes":{"components/backmatter/References.jsx":"66c60a551924","components/blocks/Admonition.jsx":"3f708466064f","components/blocks/Aside.jsx":"44569d4d06a0","components/blocks/Proof.jsx":"087a37247478","components/core/Button.jsx":"97a14c29e4c4","components/core/Icon.jsx":"8f6a48d64df0","components/frontmatter/Abstract.jsx":"6dbf45785998","components/frontmatter/FrontmatterBlock.jsx":"4fe3a9d9c115","components/media/Blockquote.jsx":"ecb39f310f85","components/media/CodeBlock.jsx":"ca1b18509084","components/media/Equation.jsx":"9a42913e3dd1","components/media/Figure.jsx":"585bd950af09","components/slide/FooterLink.jsx":"05fcc333e89e","components/slide/Slide.jsx":"f48659cc69a2","components/slide/SlideHeader.jsx":"bc06e425e291"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.ArticleSlidesDesignSystem_a47fec = window.ArticleSlidesDesignSystem_a47fec || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/backmatter/References.jsx
try { (() => {
function References({
  title = 'References',
  items = [],
  start = 1,
  style
}) {
  return /*#__PURE__*/React.createElement("section", {
    style: style
  }, /*#__PURE__*/React.createElement("header", {
    style: {
      fontSize: '1.125em',
      fontWeight: 600,
      color: 'var(--myst-color-text)',
      lineHeight: 1.6
    }
  }, title), /*#__PURE__*/React.createElement("div", {
    style: {
      paddingLeft: '0.75em',
      fontSize: 'var(--slide-fs-label)',
      lineHeight: 1.6,
      color: 'var(--myst-color-text-tertiary)'
    }
  }, /*#__PURE__*/React.createElement("ol", {
    start: start,
    style: {
      margin: '0.5em 0 0',
      paddingLeft: '1.625em'
    }
  }, items.map((it, i) => /*#__PURE__*/React.createElement("li", {
    key: i,
    style: {
      margin: '0.4em 0',
      overflowWrap: 'anywhere'
    }
  }, it)))));
}
Object.assign(__ds_scope, { References });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/backmatter/References.jsx", error: String((e && e.message) || e) }); }

// components/blocks/Aside.jsx
try { (() => {
function Aside({
  kind = 'margin',
  title,
  children,
  style
}) {
  if (kind === 'topic') {
    return /*#__PURE__*/React.createElement("aside", {
      style: {
        margin: '1.25em 0',
        boxShadow: 'var(--shadow)',
        overflow: 'hidden',
        background: 'var(--myst-color-bg)',
        ...style
      }
    }, title && /*#__PURE__*/React.createElement("div", {
      style: {
        margin: 0,
        fontWeight: 500,
        padding: '0.5em 1em',
        borderTop: '1px solid var(--myst-color-border)',
        borderBottom: '1px solid var(--myst-color-border)',
        background: 'var(--myst-color-surface)',
        color: 'var(--myst-color-text)'
      }
    }, title), /*#__PURE__*/React.createElement("div", {
      style: {
        padding: '0.5em 1em'
      }
    }, children));
  }
  return /*#__PURE__*/React.createElement("aside", {
    style: {
      fontSize: 'var(--slide-fs-small)',
      lineHeight: 'var(--slide-lh-small)',
      ...style
    }
  }, title && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 'var(--slide-fs-body)',
      lineHeight: 1.5,
      fontWeight: 600,
      color: 'var(--myst-color-text)',
      marginBottom: '0.25em'
    }
  }, title), /*#__PURE__*/React.createElement("div", null, children));
}
Object.assign(__ds_scope, { Aside });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/blocks/Aside.jsx", error: String((e && e.message) || e) }); }

// components/blocks/Proof.jsx
try { (() => {
const SCHEME = {
  proof: 'proof',
  algorithm: 'proof',
  lemma: 'theorem',
  conjecture: 'theorem',
  theorem: 'theorem',
  observation: 'warning',
  assumption: 'warning',
  axiom: 'warning',
  criterion: 'example',
  corollary: 'example',
  property: 'example',
  example: 'success',
  remark: 'danger',
  definition: 'info',
  proposition: 'info'
};
function Proof({
  kind = 'theorem',
  number,
  title,
  children,
  style
}) {
  const s = SCHEME[kind] || 'proof';
  const label = kind.slice(0, 1).toUpperCase() + kind.slice(1);
  return /*#__PURE__*/React.createElement("aside", {
    style: {
      margin: '1.25em 0',
      boxShadow: 'var(--shadow)',
      overflow: 'hidden',
      borderLeft: '0.25em solid var(--myst-color-' + s + ')',
      background: 'var(--myst-color-bg)',
      ...style
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      margin: 0,
      fontWeight: 500,
      padding: '0.5rem 0',
      display: 'flex',
      borderTop: '1px solid var(--myst-color-border)',
      borderBottom: '1px solid var(--myst-color-border)',
      background: 'var(--myst-color-' + s + '-bg)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      marginLeft: '1em',
      color: 'var(--myst-color-' + s + '-text)'
    }
  }, label, number != null && ' ' + number, title && /*#__PURE__*/React.createElement(React.Fragment, null, " (", title, ")"))), /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '0.5em 1em'
    }
  }, children));
}
Object.assign(__ds_scope, { Proof });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/blocks/Proof.jsx", error: String((e && e.message) || e) }); }

// components/core/Icon.jsx
try { (() => {
// Generated from assets/icons (Heroicons 24 outline/solid, MIT; scienceicons 24 solid). Inherits currentColor.
const ICONS = {
  "outline/arrow-left": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18\"></path>"
  },
  "outline/arrow-right-circle": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"m12.75 15 3-3m0 0-3-3m3 3h-7.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z\"></path>"
  },
  "outline/arrow-right": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3\"></path>"
  },
  "outline/document-arrow-down": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m.75 12 3 3m0 0 3-3m-3 3v-6m-1.5-9H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z\"></path>"
  },
  "outline/document": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z\"></path>"
  },
  "outline/exclamation-circle": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z\"></path>"
  },
  "outline/hashtag": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M5.25 8.25h15m-16.5 7.5h15m-1.8-13.5-3.9 19.5m-2.1-19.5-3.9 19.5\"></path>"
  },
  "outline/information-circle": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z\"></path>"
  },
  "outline/light-bulb": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M12 18v-5.25m0 0a6.01 6.01 0 0 0 1.5-.189m-1.5.189a6.01 6.01 0 0 1-1.5-.189m3.75 7.478a12.06 12.06 0 0 1-4.5 0m3.75 2.383a14.406 14.406 0 0 1-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 1 0-7.517 0c.85.493 1.509 1.333 1.509 2.316V18\"></path>"
  },
  "outline/megaphone": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M10.34 15.84c-.688-.06-1.386-.09-2.09-.09H7.5a4.5 4.5 0 1 1 0-9h.75c.704 0 1.402-.03 2.09-.09m0 9.18c.253.962.584 1.892.985 2.783.247.55.06 1.21-.463 1.511l-.657.38c-.551.318-1.26.117-1.527-.461a20.845 20.845 0 0 1-1.44-4.282m3.102.069a18.03 18.03 0 0 1-.59-4.59c0-1.586.205-3.124.59-4.59m0 9.18a23.848 23.848 0 0 1 8.835 2.535M10.34 6.66a23.847 23.847 0 0 0 8.835-2.535m0 0A23.74 23.74 0 0 0 18.795 3m.38 1.125a23.91 23.91 0 0 1 1.014 5.395m-1.014 8.855c-.118.38-.245.754-.38 1.125m.38-1.125a23.91 23.91 0 0 0 1.014-5.395m0-3.46c.495.413.811 1.035.811 1.73 0 .695-.316 1.317-.811 1.73m0-3.46a24.347 24.347 0 0 1 0 3.46\"></path>"
  },
  "outline/pencil-square": {
    "vb": "0 0 24 24",
    "fill": "none",
    "stroke": "currentColor",
    "sw": "1.5",
    "inner": "<path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10\"></path>"
  },
  "solid/bolt": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path fill-rule=\"evenodd\" d=\"M14.615 1.595a.75.75 0 0 1 .359.852L12.982 9.75h7.268a.75.75 0 0 1 .548 1.262l-10.5 11.25a.75.75 0 0 1-1.272-.71l1.992-7.302H3.75a.75.75 0 0 1-.548-1.262l10.5-11.25a.75.75 0 0 1 .913-.143Z\" clip-rule=\"evenodd\"></path>"
  },
  "solid/chevron-right": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path fill-rule=\"evenodd\" d=\"M16.28 11.47a.75.75 0 0 1 0 1.06l-7.5 7.5a.75.75 0 0 1-1.06-1.06L14.69 12 7.72 5.03a.75.75 0 0 1 1.06-1.06l7.5 7.5Z\" clip-rule=\"evenodd\"></path>"
  },
  "solid/exclamation-circle": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path fill-rule=\"evenodd\" d=\"M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12ZM12 8.25a.75.75 0 0 1 .75.75v3.75a.75.75 0 0 1-1.5 0V9a.75.75 0 0 1 .75-.75Zm0 8.25a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z\" clip-rule=\"evenodd\"></path>"
  },
  "solid/exclamation-triangle": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path fill-rule=\"evenodd\" d=\"M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003ZM12 8.25a.75.75 0 0 1 .75.75v3.75a.75.75 0 0 1-1.5 0V9a.75.75 0 0 1 .75-.75Zm0 8.25a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z\" clip-rule=\"evenodd\"></path>"
  },
  "solid/x-circle": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path fill-rule=\"evenodd\" d=\"M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25Zm-1.72 6.97a.75.75 0 1 0-1.06 1.06L10.94 12l-1.72 1.72a.75.75 0 1 0 1.06 1.06L12 13.06l1.72 1.72a.75.75 0 1 0 1.06-1.06L13.06 12l1.72-1.72a.75.75 0 1 0-1.06-1.06L12 10.94l-1.72-1.72Z\" clip-rule=\"evenodd\"></path>"
  },
  "science/email": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path d=\"M21.8 18c0 1.1-.9 2-1.9 2H4.2c-1.1 0-1.9-.9-1.9-2V9.9c0-.5.3-.7.8-.4l7.8 4.7c.7.4 1.7.4 2.4 0L21 9.5c.4-.2.8-.1.8.4V18z\"></path> <path d=\"M21.8 6c0-1.1-.9-2-1.9-2H4.2c-1.1 0-2 .9-2 2v.4c0 .5.3 1.1.8 1.3l8.5 5.1c.2.1.7.1.9 0l8.6-5c.4-.3.8-.9.8-1.3-.1-.1-.1-.5 0-.5z\"></path>"
  },
  "science/github": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path d=\"M12 2.5c-5.4 0-9.8 4.4-9.8 9.7 0 4.3 2.8 8 6.7 9.2.5.1.7-.2.7-.5v-1.8c-2.4.5-3.1-.6-3.3-1.1-.1-.3-.6-1.1-1-1.4-.3-.2-.8-.6 0-.6s1.3.7 1.5 1c.9 1.5 2.3 1.1 2.8.8.1-.6.3-1.1.6-1.3-2.2-.2-4.4-1.1-4.4-4.8 0-1.1.4-1.9 1-2.6-.1-.2-.4-1.2.1-2.6 0 0 .8-.3 2.7 1 .8-.2 1.6-.3 2.4-.3.8 0 1.7.1 2.4.3 1.9-1.3 2.7-1 2.7-1 .5 1.3.2 2.3.1 2.6.6.7 1 1.5 1 2.6 0 3.7-2.3 4.6-4.4 4.8.4.3.7.9.7 1.8V21c0 .3.2.6.7.5 3.9-1.3 6.6-4.9 6.6-9.2 0-5.4-4.4-9.8-9.8-9.8z\"></path>"
  },
  "science/open-access": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path d=\"M17.1 12.6h-2V7.5c0-1.7-1.4-3.1-3-3.1-.8 0-1.6.3-2.2.9-.6.5-.9 1.3-.9 2.2v.7H7v-.7c0-1.4.5-2.7 1.5-3.7s2.2-1.5 3.6-1.5 2.6.5 3.6 1.5 1.5 2.3 1.5 3.7v5.1z\"></path> <path d=\"M12 21.8c-.8 0-1.6-.2-2.3-.5-.7-.3-1.4-.8-1.9-1.3-.6-.6-1-1.2-1.3-2-.3-.8-.5-1.6-.5-2.4s.2-1.6.5-2.4c.3-.7.7-1.4 1.3-2s1.2-1 1.9-1.3c.7-.3 1.5-.5 2.3-.5.8 0 1.6.2 2.3.5.7.3 1.4.8 1.9 1.3.6.6 1 1.2 1.3 2 .3.8.5 1.6.5 2.4s-.2 1.6-.5 2.4c-.3.7-.7 1.4-1.3 2-.6.6-1.2 1-1.9 1.3-.7.3-1.5.5-2.3.5zm0-10.3c-2.2 0-4 1.8-4 4.1s1.8 4.1 4 4.1 4-1.8 4-4.1-1.8-4.1-4-4.1z\"></path> <circle cx=\"12\" cy=\"15.6\" r=\"1.7\"></circle>"
  },
  "science/orcid": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path d=\"M21.8 12c0 5.4-4.4 9.8-9.8 9.8S2.2 17.4 2.2 12 6.6 2.2 12 2.2s9.8 4.4 9.8 9.8zM8.2 5.8c-.4 0-.8.3-.8.8s.3.8.8.8.8-.4.8-.8-.3-.8-.8-.8zm2.3 9.6h1.2v-6h1.8c2.3 0 3.3 1.4 3.3 3s-1.5 3-3.3 3h-3v1.1H9V8.3H7.7v8.2h5.9c3.3 0 4.5-2.2 4.5-4.1s-1.2-4.1-4.3-4.1h-3.2l-.1 7.1z\"></path>"
  },
  "science/ror": {
    "vb": "0 0 24 24",
    "fill": "currentColor",
    "stroke": null,
    "sw": null,
    "inner": "<path d=\"M10 4.2L8.3 6.8 6.6 4.2H10zM17.1 4.2l-1.7 2.6-1.7-2.6h3.4zM6.6 19.8l1.7-2.6 1.7 2.6H6.6zM13.7 19.8l1.7-2.6 1.7 2.6h-3.4zM20.8 12.5c.6-.1 1.1-.4 1.4-.8.3-.4.5-.9.5-1.5 0-.5-.1-.9-.3-1.2-.2-.3-.4-.6-.7-.8-.3-.2-.6-.3-1-.4-.4-.1-.8-.1-1.2-.1h-3.3v2.6c0-.1-.1-.2-.1-.2-.2-.6-.6-1-1-1.4-.4-.4-.9-.7-1.5-.9-.6-.2-1.2-.3-1.9-.3s-1.3.1-1.9.3c-.5.1-1 .4-1.4.8-.3.4-.6.8-.9 1.3 0-.3-.1-.6-.2-.9-.2-.4-.4-.6-.7-.8-.3-.2-.6-.3-1-.4s-.8-.2-1.3-.2H1v8.5h1.9v-3.4h.9l1.8 3.4h2.3l-2.2-3.6c.6-.1 1.1-.4 1.4-.8v-.1.2c0 .7.1 1.3.3 1.8.2.6.6 1 1 1.4.4.4.9.7 1.5.9.6.2 1.2.3 1.9.3s1.3-.1 1.9-.3c.6-.2 1.1-.5 1.5-.9.4-.4.7-.9 1-1.4 0-.1.1-.2.1-.2V16H18v-3.4h.9l1.8 3.4H23l-2.2-3.5zM5.4 10.7c-.1.2-.2.3-.3.3-.2.1-.3.1-.5.1H2.9V9.2h1.7c.2 0 .3.1.5.1.1.1.3.2.3.3.1.1.1.3.1.5.1.3 0 .5-.1.6zm8.8 2.3c-.1.3-.3.6-.5.9-.2.2-.5.4-.8.6-.3.1-.7.2-1.1.2-.4 0-.8-.1-1.1-.2-.3-.1-.6-.3-.8-.6-.2-.2-.4-.5-.5-.9-.1-.3-.2-.7-.2-1.1 0-.4.1-.8.2-1.1s.3-.6.5-.9c.2-.2.5-.4.8-.6.3-.1.7-.2 1.1-.2.4 0 .8.1 1.1.2.3.1.6.3.8.6.2.2.4.5.5.9.1.3.2.7.2 1.1 0 .4 0 .7-.2 1.1zm6.4-2.3c-.1.1-.2.2-.4.3-.2.1-.3.1-.5.1H18V9.2h1.7c.2 0 .3.1.5.1.1.1.3.2.3.3.1.1.1.3.1.5.1.3.1.5 0 .6z\"></path>"
  }
};
function Icon({
  name,
  size = '1em',
  className,
  style,
  title
}) {
  const i = ICONS[name];
  if (!i) return null;
  const fill = i.fill === 'none' ? 'none' : 'currentColor';
  const stroke = i.stroke ? 'currentColor' : undefined;
  return React.createElement('svg', {
    xmlns: 'http://www.w3.org/2000/svg',
    viewBox: i.vb,
    width: size,
    height: size,
    fill,
    stroke,
    strokeWidth: i.sw || undefined,
    className,
    style: {
      display: 'inline-block',
      flex: 'none',
      ...style
    },
    'aria-hidden': title ? undefined : true,
    role: title ? 'img' : undefined,
    dangerouslySetInnerHTML: {
      __html: (title ? '<title>' + title + '</title>' : '') + i.inner.replace(/(stroke|fill)="#[0-9a-fA-F]+"/g, '$1="currentColor"')
    }
  });
}
const ICON_NAMES = Object.keys(ICONS);
Object.assign(__ds_scope, { Icon, ICON_NAMES });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Icon.jsx", error: String((e && e.message) || e) }); }

// components/blocks/Admonition.jsx
try { (() => {
const KINDS = {
  note: ['blue', 'outline/information-circle'],
  important: ['blue', 'solid/bolt'],
  seealso: ['blue', 'outline/arrow-right-circle'],
  tip: ['green', 'outline/pencil-square'],
  hint: ['green', 'outline/light-bulb'],
  attention: ['yellow', 'outline/megaphone'],
  warning: ['yellow', 'solid/exclamation-triangle'],
  caution: ['yellow', 'outline/exclamation-circle'],
  danger: ['red', 'solid/exclamation-circle'],
  error: ['red', 'solid/x-circle']
};
const C = {
  blue: 'info',
  green: 'tip',
  yellow: 'warning',
  red: 'danger'
};
const TITLES = {
  note: 'Note',
  important: 'Important',
  seealso: 'See Also',
  tip: 'Tip',
  hint: 'Hint',
  attention: 'Attention',
  warning: 'Warning',
  caution: 'Caution',
  danger: 'Danger',
  error: 'Error'
};
function Admonition({
  kind = 'note',
  title,
  color,
  simple = false,
  hideIcon = false,
  children,
  style
}) {
  const [defColor, icon] = KINDS[kind] || KINDS.note;
  const c = C[color || defColor];
  const t = title === undefined ? TITLES[kind] : title;
  return /*#__PURE__*/React.createElement("aside", {
    style: {
      margin: '1.25em 0',
      boxShadow: 'var(--shadow-md)',
      overflow: 'hidden',
      borderLeft: (simple ? 2 : 4) / 16 + 'em solid var(--myst-color-' + c + ')',
      borderRadius: simple ? 0 : 'var(--radius)',
      background: 'var(--myst-color-bg)',
      ...style
    }
  }, t && /*#__PURE__*/React.createElement("div", {
    style: {
      margin: 0,
      fontWeight: 500,
      padding: '0.25rem 0',
      display: 'flex',
      minWidth: 0,
      fontSize: simple ? '1em' : '1.125em',
      lineHeight: 1.75,
      background: simple ? 'var(--myst-color-surface)' : 'var(--myst-color-' + c + '-bg)',
      color: 'var(--myst-color-' + c + '-text)'
    }
  }, !hideIcon && /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-flex',
      paddingLeft: '0.5rem',
      marginRight: '0.5rem',
      alignSelf: 'center',
      flex: 'none',
      color: 'var(--myst-color-' + c + '-text)'
    }
  }, /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: icon,
    size: simple ? '1.5em' : '1.6em'
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      color: 'var(--myst-color-text)',
      flexGrow: 1,
      alignSelf: 'center',
      overflow: 'hidden',
      marginLeft: hideIcon ? '1rem' : 0
    }
  }, t)), /*#__PURE__*/React.createElement("div", {
    style: {
      padding: simple ? '0.5em 1em' : '0.5em 1em'
    }
  }, children));
}
Object.assign(__ds_scope, { Admonition });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/blocks/Admonition.jsx", error: String((e && e.message) || e) }); }

// components/core/Button.jsx
try { (() => {
function Button({
  variant = 'primary',
  href,
  icon,
  children,
  onClick,
  style
}) {
  const [h, setH] = React.useState(false);
  const base = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.25em',
    textDecoration: 'none',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
    fontFamily: 'var(--font-sans)',
    fontSize: 'inherit',
    lineHeight: 1.5,
    transition: 'background-color 150ms, color 150ms'
  };
  const v = variant === 'chip' ? {
    padding: '0.25em 0.5em',
    fontWeight: 400,
    borderRadius: 'var(--radius)',
    border: '1px solid ' + (h ? 'transparent' : 'var(--ats-color-chip-border)'),
    background: h ? 'var(--ats-color-chip-hover)' : 'var(--ats-color-chip-bg)',
    color: h ? '#fff' : 'var(--myst-color-text)'
  } : {
    marginTop: '0.25rem',
    padding: '0.25em 1em',
    fontWeight: 700,
    borderRadius: 'var(--radius)',
    border: 0,
    background: h ? 'var(--myst-color-primary-hover)' : 'var(--myst-color-primary)',
    color: '#fff'
  };
  const Tag = href ? 'a' : 'button';
  return /*#__PURE__*/React.createElement(Tag, {
    href: href,
    onClick: onClick,
    onMouseEnter: () => setH(true),
    onMouseLeave: () => setH(false),
    style: {
      ...base,
      ...v,
      ...style
    }
  }, icon && /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: icon,
    size: "1em"
  }), /*#__PURE__*/React.createElement("span", null, children));
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Button.jsx", error: String((e && e.message) || e) }); }

// components/frontmatter/Abstract.jsx
try { (() => {
function Abstract({
  title = 'Abstract',
  keywords,
  children,
  style
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: style
  }, /*#__PURE__*/React.createElement("h2", {
    style: {
      marginBottom: '0.75em',
      fontSize: '1em',
      lineHeight: 1.5,
      fontWeight: 600
    }
  }, title), /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '0.25em 1.5em',
      marginBottom: '0.75em',
      borderRadius: '0.125rem',
      background: 'var(--myst-color-bg-secondary)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '0.75em 0'
    }
  }, children)), keywords && keywords.length > 0 && /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: '1em'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      marginRight: '0.5em',
      fontWeight: 600,
      color: 'var(--myst-color-text)'
    }
  }, "Keywords:"), /*#__PURE__*/React.createElement("span", null, keywords.join(', '))));
}
Object.assign(__ds_scope, { Abstract });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/frontmatter/Abstract.jsx", error: String((e && e.message) || e) }); }

// components/frontmatter/FrontmatterBlock.jsx
try { (() => {
const iconLink = {
  marginLeft: '0.25em',
  color: 'var(--myst-color-text-tertiary)',
  display: 'inline-flex',
  transform: 'translateY(-0.1em)',
  verticalAlign: 'middle'
};
function Author({
  a
}) {
  return /*#__PURE__*/React.createElement("span", {
    style: {
      fontWeight: 600,
      fontSize: '0.875em',
      color: 'var(--myst-color-text)'
    }
  }, /*#__PURE__*/React.createElement("span", null, a.name), a.email && a.corresponding && /*#__PURE__*/React.createElement("a", {
    href: 'mailto:' + a.email,
    title: a.email,
    style: iconLink
  }, /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: "science/email",
    size: "1.1429em"
  })), a.orcid && /*#__PURE__*/React.createElement("a", {
    href: 'https://orcid.org/' + a.orcid,
    title: 'ORCID: ' + a.orcid,
    style: iconLink
  }, /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: "science/orcid",
    size: "1.1429em"
  })));
}
function FrontmatterBlock({
  subject,
  venue,
  volume,
  issue,
  title,
  subtitle,
  authors = [],
  affiliations = [],
  date,
  doi,
  openAccess,
  github,
  style
}) {
  const aff = Object.fromEntries(affiliations.map(x => [x.id, x]));
  const hasAff = authors.some(a => a.affiliations && a.affiliations.length);
  const hasHeader = subject || venue || openAccess || github;
  const doiUrl = doi ? 'https://doi.org/' + doi.replace(/^(https?:\/\/)?(dx\.)?doi\.org\//, '') : null;
  const dateStr = date ? new Date(date + 'T00:00:00').toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  }) : null;
  return /*#__PURE__*/React.createElement("div", {
    style: style
  }, hasHeader && /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      marginBottom: '1.25em',
      height: '1.5em',
      fontSize: '0.75em',
      fontWeight: 300
    }
  }, subject && /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 'none',
      paddingRight: '0.5em',
      fontVariant: 'small-caps',
      ...(venue ? {
        marginRight: '0.5em',
        borderRight: '1px solid var(--myst-color-border)'
      } : null)
    }
  }, subject), venue && /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 'none',
      marginRight: '0.5em'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontWeight: 600,
      fontVariant: 'small-caps'
    }
  }, venue), volume != null && /*#__PURE__*/React.createElement("span", {
    style: {
      paddingLeft: '0.5em',
      marginLeft: '0.5em',
      borderLeft: '1px solid var(--myst-color-border)'
    }
  }, "Volume ", volume, issue != null && /*#__PURE__*/React.createElement(React.Fragment, null, ", Issue ", issue))), /*#__PURE__*/React.createElement("div", {
    style: {
      flexGrow: 1
    }
  }), openAccess && /*#__PURE__*/React.createElement("span", {
    title: "Open Access",
    style: {
      opacity: 0.6,
      marginRight: '0.25em',
      display: 'inline-flex'
    }
  }, /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: "science/open-access",
    size: "1.6667em"
  })), github && /*#__PURE__*/React.createElement("span", {
    title: 'GitHub: ' + github,
    style: {
      opacity: 0.6,
      marginRight: '0.25em',
      display: 'inline-flex'
    }
  }, /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: "science/github",
    size: "1.6667em"
  }))), title && /*#__PURE__*/React.createElement("h1", {
    style: {
      marginBottom: 0
    }
  }, title), subtitle && /*#__PURE__*/React.createElement("p", {
    className: "lead",
    style: {
      marginTop: '0.4em',
      marginBottom: 0,
      color: 'var(--myst-color-text-tertiary)'
    }
  }, subtitle), authors.length > 0 && /*#__PURE__*/React.createElement("header", {
    style: {
      marginTop: '1.5em'
    }
  }, hasAff ? /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      rowGap: '0.25em',
      columnGap: '1em'
    }
  }, authors.length > 1 && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    style: labelStyle
  }, "Authors"), /*#__PURE__*/React.createElement("div", {
    style: labelStyle
  }, "Affiliations")), authors.map(a => /*#__PURE__*/React.createElement(React.Fragment, {
    key: a.name
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement(Author, {
    a: a
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.875em'
    }
  }, (a.affiliations || []).map(id => /*#__PURE__*/React.createElement("div", {
    key: id
  }, aff[id] && aff[id].name || id)))))) : /*#__PURE__*/React.createElement("div", null, authors.map((a, i) => /*#__PURE__*/React.createElement("span", {
    key: a.name
  }, /*#__PURE__*/React.createElement(Author, {
    a: a
  }), i < authors.length - 1 && /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.875em'
    }
  }, ",\xA0"))))), (dateStr || doiUrl) && /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      marginTop: '1em',
      fontSize: '0.75em',
      fontWeight: 300
    }
  }, dateStr && /*#__PURE__*/React.createElement("time", {
    dateTime: date
  }, dateStr), dateStr && doiUrl && /*#__PURE__*/React.createElement("span", {
    style: {
      margin: '0 0.5em'
    }
  }, "\xB7"), doiUrl && /*#__PURE__*/React.createElement("a", {
    href: doiUrl,
    style: {
      color: 'inherit',
      textDecoration: 'none',
      fontWeight: 300
    }
  }, doiUrl)));
}
const labelStyle = {
  paddingBottom: '0.5em',
  fontSize: '0.75em',
  fontWeight: 100,
  textTransform: 'uppercase',
  letterSpacing: '0.02em'
};
Object.assign(__ds_scope, { FrontmatterBlock });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/frontmatter/FrontmatterBlock.jsx", error: String((e && e.message) || e) }); }

// components/media/Blockquote.jsx
try { (() => {
function Blockquote({
  children,
  attribution,
  pull = false,
  style
}) {
  return /*#__PURE__*/React.createElement("figure", {
    style: {
      margin: '1.6em 0',
      ...style
    }
  }, /*#__PURE__*/React.createElement("blockquote", {
    style: {
      margin: 0,
      fontWeight: 500,
      fontStyle: 'italic',
      color: 'var(--myst-color-text)',
      borderLeft: '0.25rem solid var(--myst-color-border)',
      paddingLeft: '1em',
      fontSize: pull ? '1.4em' : '1em',
      lineHeight: pull ? 1.5 : 1.75
    }
  }, children), attribution && /*#__PURE__*/React.createElement("figcaption", {
    style: {
      textAlign: 'right',
      marginTop: '0.75em',
      fontSize: '0.875em',
      color: 'var(--myst-color-text-tertiary)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      paddingRight: '0.5em'
    }
  }, "\u2014"), attribution));
}
Object.assign(__ds_scope, { Blockquote });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/media/Blockquote.jsx", error: String((e && e.message) || e) }); }

// components/media/CodeBlock.jsx
try { (() => {
function CodeBlock({
  code = '',
  filename,
  executable = false,
  showLineNumbers = false,
  startingLineNumber = 1,
  emphasizeLines = [],
  style
}) {
  const lines = code.replace(/\n$/, '').split('\n');
  const hl = new Set(emphasizeLines);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative',
      margin: '1.25em 0',
      fontSize: '0.875em',
      boxShadow: 'var(--shadow)',
      background: executable ? 'var(--myst-color-bg)' : 'rgb(231 229 228 / 0.1)',
      border: executable ? '1px solid var(--myst-color-border)' : 'none',
      borderLeft: executable ? '4px solid var(--myst-color-link-underline)' : undefined,
      ...style
    }
  }, filename && /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      paddingLeft: '0.5em',
      background: 'var(--myst-color-bg-secondary)',
      borderBottom: '1px solid var(--myst-color-border)'
    }
  }, /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: "outline/document",
    size: "1em",
    style: {
      color: 'var(--myst-color-text-tertiary)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '0.5em',
      fontSize: '0.875em',
      lineHeight: 1,
      color: 'var(--myst-color-text-secondary)',
      fontFamily: 'var(--font-sans)'
    }
  }, filename)), /*#__PURE__*/React.createElement("pre", {
    style: {
      margin: 0,
      padding: '0.75em',
      overflow: 'auto',
      fontFamily: 'var(--font-mono)',
      lineHeight: 1.6,
      color: 'var(--myst-color-text)'
    }
  }, lines.map((l, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    style: {
      background: hl.has(i + startingLineNumber) ? 'var(--myst-color-active-bg)' : undefined,
      borderLeft: hl.has(i + startingLineNumber) ? '4px solid var(--myst-color-active)' : '4px solid transparent',
      marginLeft: '-0.75em',
      paddingLeft: 'calc(0.75em - 4px)'
    }
  }, showLineNumbers && /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-block',
      minWidth: '1.25em',
      paddingRight: '1em',
      textAlign: 'right',
      userSelect: 'none',
      color: 'var(--myst-color-text-tertiary)'
    }
  }, i + startingLineNumber), l || '\u00a0'))));
}
Object.assign(__ds_scope, { CodeBlock });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/media/CodeBlock.jsx", error: String((e && e.message) || e) }); }

// components/media/Equation.jsx
try { (() => {
function Equation({
  tex,
  number,
  children,
  style
}) {
  const [html, setHtml] = React.useState(null);
  React.useEffect(() => {
    if (!tex) return;
    let tries = 0,
      t;
    const go = () => {
      if (window.katex) {
        try {
          setHtml(window.katex.renderToString(tex, {
            displayMode: true,
            throwOnError: false
          }));
        } catch (e) {}
        return;
      }
      if (tries++ < 40) t = setTimeout(go, 100);
    };
    go();
    return () => clearTimeout(t);
  }, [tex]);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      margin: '1em 0',
      ...style
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      flexGrow: 1,
      minWidth: 0,
      textAlign: 'center',
      overflowX: 'auto',
      color: 'var(--myst-color-text)',
      fontSize: '1.1em'
    }
  }, html ? /*#__PURE__*/React.createElement("div", {
    className: "katex-display",
    style: {
      margin: 0
    },
    dangerouslySetInnerHTML: {
      __html: html
    }
  }) : children || /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-serif)',
      fontStyle: 'italic'
    }
  }, tex)), number != null && /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 'none',
      paddingLeft: '1em',
      color: 'var(--myst-color-text)',
      userSelect: 'none'
    }
  }, "(", number, ")"));
}
Object.assign(__ds_scope, { Equation });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/media/Equation.jsx", error: String((e && e.message) || e) }); }

// components/media/Figure.jsx
try { (() => {
function Figure({
  src,
  alt = '',
  label = 'Figure',
  number,
  caption,
  ratio = '16 / 9',
  fit = 'contain',
  align = 'left',
  children,
  style
}) {
  const media = children || (src ? /*#__PURE__*/React.createElement("img", {
    src: src,
    alt: alt,
    style: {
      display: 'block',
      width: '100%',
      aspectRatio: ratio,
      objectFit: fit,
      margin: 0
    }
  }) : /*#__PURE__*/React.createElement("div", {
    style: {
      aspectRatio: ratio,
      width: '100%',
      background: 'var(--myst-color-bg-secondary)',
      border: '1px dashed var(--myst-color-border-strong)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'var(--myst-color-text-tertiary)',
      fontSize: 'var(--slide-fs-label)'
    }
  }, alt || 'Image'));
  return /*#__PURE__*/React.createElement("figure", {
    style: {
      margin: 0,
      ...style
    }
  }, media, (caption || number != null) && /*#__PURE__*/React.createElement("figcaption", {
    style: {
      marginTop: '0.857em',
      fontSize: '0.875em',
      lineHeight: 1.714,
      color: 'var(--myst-color-text-tertiary)',
      textAlign: align
    }
  }, number != null && /*#__PURE__*/React.createElement("span", {
    style: {
      fontWeight: 700,
      marginRight: '0.25em',
      color: 'var(--myst-color-text-secondary)'
    }
  }, label, " ", number, ":"), caption));
}
Object.assign(__ds_scope, { Figure });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/media/Figure.jsx", error: String((e && e.message) || e) }); }

// components/slide/FooterLink.jsx
try { (() => {
function FooterLink({
  title,
  group,
  href = '#',
  direction = 'next',
  style
}) {
  const [h, setH] = React.useState(false);
  const prev = direction === 'prev';
  return /*#__PURE__*/React.createElement("a", {
    href: href,
    onMouseEnter: () => setH(true),
    onMouseLeave: () => setH(false),
    style: {
      flex: 1,
      display: 'block',
      padding: '1em',
      fontWeight: 400,
      textDecoration: 'none',
      color: h ? 'var(--myst-color-active)' : 'var(--myst-color-text-secondary)',
      border: '1px solid',
      borderColor: h ? 'var(--myst-color-active)' : 'var(--myst-color-border)',
      borderRadius: 'var(--radius)',
      boxShadow: h ? '0 10px 15px -3px rgb(0 0 0 / 0.1),0 4px 6px -4px rgb(0 0 0 / 0.1)' : '0 1px 2px 0 rgb(0 0 0 / 0.05)',
      transition: 'all 150ms',
      ...style
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.5em'
    }
  }, prev && /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: "outline/arrow-left",
    size: "1.5em",
    style: {
      transform: h ? 'translateX(-0.25em)' : 'none',
      transition: 'transform 150ms'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      flexGrow: 1,
      textAlign: prev ? 'right' : 'left'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: '0.75em',
      color: 'var(--myst-color-text-tertiary)'
    }
  }, group || '\u00a0'), title), !prev && /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: "outline/arrow-right",
    size: "1.5em",
    style: {
      transform: h ? 'translateX(0.25em)' : 'none',
      transition: 'transform 150ms'
    }
  })));
}
Object.assign(__ds_scope, { FooterLink });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/slide/FooterLink.jsx", error: String((e && e.message) || e) }); }

// components/slide/Slide.jsx
try { (() => {
const COLS = {
  body: 'var(--slide-body-w)',
  'body-margin': 'var(--slide-body-w) var(--slide-gutter) var(--slide-margin-w)',
  page: 'minmax(0,1fr)',
  center: 'var(--slide-body-w)',
  screen: 'minmax(0,1fr)'
};
function Slide({
  layout = 'body',
  header,
  margin,
  footer,
  children,
  align = 'start',
  dark = false,
  background,
  label,
  style,
  className
}) {
  const isScreen = layout === 'screen';
  const isCenter = layout === 'center';
  return /*#__PURE__*/React.createElement("section", {
    "data-screen-label": label,
    className: ['ats-prose', dark ? 'dark' : '', className].filter(Boolean).join(' '),
    style: {
      position: 'relative',
      width: 'var(--slide-w)',
      height: 'var(--slide-h)',
      overflow: 'hidden',
      boxSizing: 'border-box',
      background: background || 'var(--myst-color-bg)',
      display: 'flex',
      flexDirection: 'column',
      padding: isScreen ? 0 : 'var(--slide-pad-top) var(--slide-pad-x) var(--slide-pad-bottom)',
      ...style
    }
  }, header && /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 'none',
      marginBottom: 56,
      ...(isScreen ? {
        position: 'absolute',
        top: 'var(--slide-pad-top)',
        left: 'var(--slide-pad-x)',
        right: 'var(--slide-pad-x)',
        zIndex: 2,
        marginBottom: 0
      } : null)
    }
  }, header), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1,
      minHeight: 0,
      display: 'grid',
      gridTemplateColumns: COLS[layout] || COLS.body,
      justifyContent: isCenter ? 'center' : 'start',
      alignContent: isCenter ? 'center' : align === 'center' ? 'center' : align === 'end' ? 'end' : 'start'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      minWidth: 0
    }
  }, children), layout === 'body-margin' && /*#__PURE__*/React.createElement("div", null), layout === 'body-margin' && /*#__PURE__*/React.createElement("div", {
    style: {
      minWidth: 0,
      fontSize: 'var(--slide-fs-small)',
      lineHeight: 'var(--slide-lh-small)'
    }
  }, margin)), footer && /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 'none',
      marginTop: 40
    }
  }, footer));
}
Object.assign(__ds_scope, { Slide });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/slide/Slide.jsx", error: String((e && e.message) || e) }); }

// components/slide/SlideHeader.jsx
try { (() => {
function SlideHeader({
  subject,
  venue,
  volume,
  issue,
  number,
  total,
  right,
  style
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      height: '1.5em',
      fontSize: 'var(--slide-fs-label)',
      fontWeight: 300,
      color: 'var(--myst-color-prose-body)',
      lineHeight: 1.5,
      ...style
    }
  }, subject && /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 'none',
      paddingRight: '0.5em',
      fontVariant: 'small-caps',
      ...(venue ? {
        marginRight: '0.5em',
        borderRight: '1px solid var(--myst-color-border)'
      } : null)
    }
  }, subject), venue && /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 'none',
      marginRight: '0.5em'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontWeight: 600,
      fontVariant: 'small-caps'
    }
  }, venue), volume != null && /*#__PURE__*/React.createElement("span", {
    style: {
      paddingLeft: '0.5em',
      marginLeft: '0.5em',
      borderLeft: '1px solid var(--myst-color-border)'
    }
  }, "Volume ", volume, issue != null && /*#__PURE__*/React.createElement(React.Fragment, null, ", Issue ", issue))), /*#__PURE__*/React.createElement("div", {
    style: {
      flexGrow: 1
    }
  }), right, number != null && /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 'none',
      marginLeft: '1em',
      fontVariantNumeric: 'tabular-nums',
      color: 'var(--myst-color-text-tertiary)'
    }
  }, number, total != null && /*#__PURE__*/React.createElement(React.Fragment, null, " / ", total)));
}
Object.assign(__ds_scope, { SlideHeader });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/slide/SlideHeader.jsx", error: String((e && e.message) || e) }); }

__ds_ns.References = __ds_scope.References;

__ds_ns.Admonition = __ds_scope.Admonition;

__ds_ns.Aside = __ds_scope.Aside;

__ds_ns.Proof = __ds_scope.Proof;

__ds_ns.Button = __ds_scope.Button;

__ds_ns.Icon = __ds_scope.Icon;

__ds_ns.ICON_NAMES = __ds_scope.ICON_NAMES;

__ds_ns.Abstract = __ds_scope.Abstract;

__ds_ns.FrontmatterBlock = __ds_scope.FrontmatterBlock;

__ds_ns.Blockquote = __ds_scope.Blockquote;

__ds_ns.CodeBlock = __ds_scope.CodeBlock;

__ds_ns.Equation = __ds_scope.Equation;

__ds_ns.Figure = __ds_scope.Figure;

__ds_ns.FooterLink = __ds_scope.FooterLink;

__ds_ns.Slide = __ds_scope.Slide;

__ds_ns.SlideHeader = __ds_scope.SlideHeader;

})();
