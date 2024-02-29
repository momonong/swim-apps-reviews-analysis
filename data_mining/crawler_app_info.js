var store = require('app-store-scraper');

store.app({id: 1548487050}).then(console.log).catch(console.log);

console.log('Crawling app info...')