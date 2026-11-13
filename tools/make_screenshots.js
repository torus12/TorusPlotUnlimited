#!/usr/bin/env node
/*
 * Capture the README / user-guide screenshots from the running TorusPlot app.
 * Developer tool only (needs Node.js + Playwright with Chromium); not used by the app.
 *
 *   python -m streamlit run app/streamlit_app.py --server.port 8765 --server.headless true &
 *   node tools/make_screenshots.js docs/images http://localhost:8765
 *
 * Every screenshot shows the real app working on the bundled synthetic sample data.
 */
const { chromium } = require('playwright');

const out = process.argv[2] || 'docs/images';
const url = process.argv[3] || 'http://localhost:8765';

async function nav(page, label, marker) {
  await page.locator('section[data-testid="stSidebar"] label').filter({ hasText: label }).click();
  await page.waitForSelector(`text=${marker}`, { timeout: 60000 });
  await page.waitForTimeout(1200);
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForSelector('text=Load a sample dataset', { timeout: 90000 });
  await page.screenshot({ path: `${out}/app_data.png` });

  await page.getByRole('button', { name: 'Load sample' }).click();
  await page.waitForSelector('text=Loaded Column table', { timeout: 60000 });

  await nav(page, '2 Analyze', 'Not sure which test');
  await page.getByRole('button', { name: 'Recommend a test' }).click();
  await page.waitForSelector('text=Recommended:', { timeout: 60000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${out}/app_analyze.png` });
  await page.getByRole('button', { name: /^Run: / }).click();
  await page.waitForSelector('text=Assumption checks', { timeout: 60000 });
  const results = page.getByText('Results for this table');
  await results.scrollIntoViewIfNeeded();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${out}/app_results.png` });

  await nav(page, '3 Graph', 'New figure');
  await page.getByRole('button', { name: 'Create figure' }).click();
  await page.waitForSelector('text=Live preview', { timeout: 60000 });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${out}/app_graph.png` });

  await nav(page, '4 Style', 'Style preset');
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${out}/app_style.png` });

  await nav(page, '5 Compose', 'Panels (in reading order)');
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${out}/app_compose.png` });

  await nav(page, '6 Export', 'What to export');
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${out}/app_export.png` });

  await nav(page, 'AI Assistant', 'Describe it');
  await page.getByRole('tab', { name: 'Match this figure' }).click();
  await page.waitForTimeout(800);
  // style source: one of TorusPlot's own gallery images (classic black/grey look)
  await page.locator('input[type="file"]').nth(0).setInputFiles(`${out}/plot_violin.png`);
  await page.waitForTimeout(1500);
  await page.getByRole('button', { name: 'Match this figure' }).click();
  await page.waitForSelector('text=Colours measured from the reference', { timeout: 90000 });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${out}/app_assistant.png` });

  await browser.close();
  console.log(`screenshots written to ${out}`);
})().catch((err) => { console.error(err); process.exit(1); });
