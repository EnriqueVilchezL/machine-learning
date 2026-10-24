import React from 'react';
import { Icon } from '../core/Icon.jsx';

const iconLink = { marginLeft: '0.25em', color: 'var(--myst-color-text-tertiary)', display: 'inline-flex', transform: 'translateY(-0.1em)', verticalAlign: 'middle' };

function Author({ a }) {
  return (
    <span style={{ fontWeight: 600, fontSize: '0.875em', color: 'var(--myst-color-text)' }}>
      <span>{a.name}</span>
      {a.email && a.corresponding && <a href={'mailto:' + a.email} title={a.email} style={iconLink}><Icon name="science/email" size="1.1429em" /></a>}
      {a.orcid && <a href={'https://orcid.org/' + a.orcid} title={'ORCID: ' + a.orcid} style={iconLink}><Icon name="science/orcid" size="1.1429em" /></a>}
    </span>
  );
}

export function FrontmatterBlock({ subject, venue, volume, issue, title, subtitle, authors = [], affiliations = [], date, doi, openAccess, github, style }) {
  const aff = Object.fromEntries(affiliations.map((x) => [x.id, x]));
  const hasAff = authors.some((a) => a.affiliations && a.affiliations.length);
  const hasHeader = subject || venue || openAccess || github;
  const doiUrl = doi ? 'https://doi.org/' + doi.replace(/^(https?:\/\/)?(dx\.)?doi\.org\//, '') : null;
  const dateStr = date ? new Date(date + 'T00:00:00').toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }) : null;
  return (
    <div style={style}>
      {hasHeader && (
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.25em', height: '1.5em', fontSize: '0.75em', fontWeight: 300 }}>
          {subject && <div style={{ flex: 'none', paddingRight: '0.5em', fontVariant: 'small-caps', ...(venue ? { marginRight: '0.5em', borderRight: '1px solid var(--myst-color-border)' } : null) }}>{subject}</div>}
          {venue && <div style={{ flex: 'none', marginRight: '0.5em' }}><span style={{ fontWeight: 600, fontVariant: 'small-caps' }}>{venue}</span>{volume != null && <span style={{ paddingLeft: '0.5em', marginLeft: '0.5em', borderLeft: '1px solid var(--myst-color-border)' }}>Volume {volume}{issue != null && <>, Issue {issue}</>}</span>}</div>}
          <div style={{ flexGrow: 1 }} />
          {openAccess && <span title="Open Access" style={{ opacity: 0.6, marginRight: '0.25em', display: 'inline-flex' }}><Icon name="science/open-access" size="1.6667em" /></span>}
          {github && <span title={'GitHub: ' + github} style={{ opacity: 0.6, marginRight: '0.25em', display: 'inline-flex' }}><Icon name="science/github" size="1.6667em" /></span>}
        </div>
      )}
      {title && <h1 style={{ marginBottom: 0 }}>{title}</h1>}
      {subtitle && <p className="lead" style={{ marginTop: '0.4em', marginBottom: 0, color: 'var(--myst-color-text-tertiary)' }}>{subtitle}</p>}
      {authors.length > 0 && (
        <header style={{ marginTop: '1.5em' }}>
          {hasAff ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', rowGap: '0.25em', columnGap: '1em' }}>
              {authors.length > 1 && <><div style={labelStyle}>Authors</div><div style={labelStyle}>Affiliations</div></>}
              {authors.map((a) => (
                <React.Fragment key={a.name}>
                  <div><Author a={a} /></div>
                  <div style={{ fontSize: '0.875em' }}>{(a.affiliations || []).map((id) => <div key={id}>{(aff[id] && aff[id].name) || id}</div>)}</div>
                </React.Fragment>
              ))}
            </div>
          ) : (
            <div>{authors.map((a, i) => <span key={a.name}><Author a={a} />{i < authors.length - 1 && <span style={{ fontSize: '0.875em' }}>,&nbsp;</span>}</span>)}</div>
          )}
        </header>
      )}
      {(dateStr || doiUrl) && (
        <div style={{ display: 'flex', marginTop: '1em', fontSize: '0.75em', fontWeight: 300 }}>
          {dateStr && <time dateTime={date}>{dateStr}</time>}
          {dateStr && doiUrl && <span style={{ margin: '0 0.5em' }}>·</span>}
          {doiUrl && <a href={doiUrl} style={{ color: 'inherit', textDecoration: 'none', fontWeight: 300 }}>{doiUrl}</a>}
        </div>
      )}
    </div>
  );
}

const labelStyle = { paddingBottom: '0.5em', fontSize: '0.75em', fontWeight: 100, textTransform: 'uppercase', letterSpacing: '0.02em' };
