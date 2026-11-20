// Synthetic-only responsive/keyboard test for the multi-crop application controls.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('playwright');
test('multi-crop controls work by keyboard and fit narrow mobile screens',async()=>{
 const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
 try {
  const page=await browser.newPage({viewport:{width:390,height:844}});let payload;
  await page.route('http://127.0.0.1:9999/**',async route=>{
   const url=new URL(route.request().url());
   if(url.pathname==='/api/status')return route.fulfill({json:{live_available:true,collection:{version:'SYNTHETIC',passage_count:10,crop_counts:{wheat:2,barley:2,maize:2,chickpea:2,mustard:2},curation:'AI-assisted fixture, not verified'}}});
   if(url.pathname==='/api/retrieve'){payload=route.request().postDataJSON();return route.fulfill({json:{mode:'local_e5',results:[]}});}
   const file={'/':'index.html','/app.js':'app.js','/style.css':'style.css','/field-hero.jpg':'field-hero.jpg'}[url.pathname];
   if(!file)return route.abort();
   return route.fulfill({body:fs.readFileSync(path.join(__dirname,'../src/krishisaarthi/ui/static',file)),contentType:file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':file.endsWith('.jpg')?'image/jpeg':'text/html'});
  });
  await page.goto('http://127.0.0.1:9999/');
  await page.getByText('SYNTHETIC · 10 passages',{exact:false}).waitFor();
  for(const width of [390,320]){
   await page.setViewportSize({width,height:844});
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  }
  await page.locator('#crop').selectOption('chickpea');
  await page.locator('#language').selectOption('hinglish');
  await page.locator('#question').fill('SYNTHETIC query');await page.locator('#question').press('Enter');
  await page.getByText('No passages found',{exact:true}).waitFor();
  assert.equal(payload.crop,'chickpea');assert.equal(payload.language,'hinglish');
  assert.equal(await page.locator('#submit').isEnabled(),true);
 } finally {await browser.close();}
});
