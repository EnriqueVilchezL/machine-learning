Code listing as MyST renders `{code}` and notebook cells.

```jsx
<CodeBlock filename="analysis.py" showLineNumbers emphasizeLines={[2]} code={'import numpy as np\nx = np.linspace(0, 1)\n'} />
<CodeBlock executable code="df.describe()" />
```

Keep listings to ~10 lines per slide. Syntax highlighting is intentionally omitted (monochrome text).
