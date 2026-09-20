const promoStyles = document.createElement('link');
promoStyles.rel = 'stylesheet';
promoStyles.href = '/fumiori-ai-ime/promo.css?v=20260906-3';
document.head.appendChild(promoStyles);

const examples = {
  distance: {
    sentence: 'レーザーで壁までの距離を<span>はかった</span>。',
    candidates: ['測った', '計った', '量った', '図った'],
  },
  time: {
    sentence: '駅まで何分かかるか時間を<span>はかった</span>。',
    candidates: ['計った', '測った', '量った', '図った'],
  },
  weight: {
    sentence: '発送前に荷物の重さを<span>はかった</span>。',
    candidates: ['量った', '測った', '計った', '図った'],
  },
  improve: {
    sentence: '工程を見直し、効率化を<span>はかった</span>。',
    candidates: ['図った', '測った', '計った', '量った'],
  },
  friend: {
    sentence: '駅前で旧友に<span>あって</span>話し込んだ。',
    candidates: ['会って', '遭って', '合って'],
  },
  accident: {
    sentence: '帰宅途中に事故に<span>あって</span>しまった。',
    candidates: ['遭って', '会って', '合って'],
  },
  match: {
    sentence: '二人の計算結果は<span>あって</span>いる。',
    candidates: ['合って', '会って', '遭って'],
  },
  law: {
    sentence: '会社は重大な法令違反を<span>おかした</span>。',
    candidates: ['犯した', '冒した', '侵した'],
  },
  risk: {
    sentence: '危険を<span>おかして</span>救助に向かった。',
    candidates: ['冒して', '犯して', '侵して'],
  },
  territory: {
    sentence: '他国の領海を<span>おかした</span>。',
    candidates: ['侵した', '犯した', '冒した'],
  },
  share: {
    sentence: '国内売上が全体の半分を<span>しめて</span>いる。',
    candidates: ['占めて', '締めて', '閉めて'],
  },
  tie: {
    sentence: '出発前にネクタイを<span>しめて</span>鏡を見た。',
    candidates: ['締めて', '閉めて', '占めて'],
  },
  door: {
    sentence: '冷房中なのでドアを<span>しめて</span>ください。',
    candidates: ['閉めて', '締めて', '占めて'],
  },
  light: {
    sentence: '暗くなったので明かりを<span>つけて</span>ください。',
    candidates: ['点けて', '付けて', '着けて'],
  },
  mask: {
    sentence: '診察室ではマスクを<span>つけて</span>ください。',
    candidates: ['着けて', '付けて', '点けて'],
  },
  tag: {
    sentence: '受付で胸に名札を<span>つけて</span>もらった。',
    candidates: ['付けて', '着けて', '点けて'],
  },
  photo: {
    sentence: '夕焼けの写真を<span>とった</span>。',
    candidates: ['撮った', '取った', '捕った'],
  },
  license: {
    sentence: '大学在学中に資格を<span>とった</span>。',
    candidates: ['取った', '撮った', '捕った'],
  },
  fish: {
    sentence: '川で大きな魚を<span>とった</span>。',
    candidates: ['捕った', '取った', '撮った'],
  },
  gpu: {
    sentence: 'DirectMLでモデルの<span>すいろん</span>を高速化した。',
    candidates: ['推論', '水論'],
  },
  quant: {
    sentence: 'モデルをINT8に<span>りょうしか</span>して軽くした。',
    candidates: ['量子化', '漁師か'],
  },
};

const sentence = document.querySelector('#demoSentence');
const stack = document.querySelector('#candidateStack');
const tabs = document.querySelectorAll('.demo-tab');
const cases = document.querySelectorAll('.demo-case');

function renderExample(key) {
  const example = examples[key];
  if (!example || !sentence || !stack) return;

  sentence.innerHTML = example.sentence;
  stack.innerHTML = example.candidates.map((candidate, index) => `
    <div class="candidate ${index === 0 ? 'selected' : ''}">
      <span class="rank">${index + 1}</span>
      <${index === 0 ? 'strong' : 'span'}>${candidate}</${index === 0 ? 'strong' : 'span'}>
      ${index === 0 ? '<span class="match">文脈に一致</span>' : ''}
    </div>
  `).join('');

  tabs.forEach((item) => {
    const active = item.dataset.example === key;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', active ? 'true' : 'false');
  });

  cases.forEach((item) => {
    item.classList.toggle('active', item.dataset.example === key);
  });
}

tabs.forEach((tab) => {
  tab.addEventListener('click', () => renderExample(tab.dataset.example));
});

cases.forEach((card) => {
  card.addEventListener('click', () => {
    renderExample(card.dataset.example);
    const demo = document.querySelector('#demo');
    if (demo) demo.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
});

const reveals = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  reveals.forEach((item) => observer.observe(item));
} else {
  reveals.forEach((item) => item.classList.add('visible'));
}

// Keep transparency and OSS information reachable from the public site without
// duplicating the full policy text inside the promotional landing page.
const policyUrl = '/fumiori-ai-ime/oss.html';
const navLinks = document.querySelector('.nav-links');
if (navLinks && !navLinks.querySelector('[data-policy-link]')) {
  const policyLink = document.createElement('a');
  policyLink.href = policyUrl;
  policyLink.textContent = 'OSS / Policies';
  policyLink.dataset.policyLink = 'true';
  const githubLink = navLinks.querySelector('.nav-github');
  navLinks.insertBefore(policyLink, githubLink || null);
}

const privacyLinks = document.querySelector('#privacy .inline-links');
if (privacyLinks && !privacyLinks.querySelector('[data-policy-link]')) {
  const policyLink = document.createElement('a');
  policyLink.href = `${policyUrl}#privacy`;
  policyLink.textContent = '日本語 / English ポリシー →';
  policyLink.dataset.policyLink = 'true';
  privacyLinks.prepend(policyLink);
}

const footerLinks = document.querySelector('.footer-links');
if (footerLinks && !footerLinks.querySelector('[data-policy-link]')) {
  const policyLink = document.createElement('a');
  policyLink.href = policyUrl;
  policyLink.textContent = 'OSS / Policies';
  policyLink.dataset.policyLink = 'true';
  footerLinks.appendChild(policyLink);
}