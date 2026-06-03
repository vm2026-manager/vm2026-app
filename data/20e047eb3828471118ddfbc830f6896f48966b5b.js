(function(window) {

  
  window.__ServerTime__ = '2026-06-02T19:34:44+0200'

  window.__SGD__ = window.__SGD__ || {};
  window.__SGD__.webapp = window.__SGD__.webapp || {};

  // Ansible migration / Cue configuration
// Mounted in by kubernetes
window.__SGD__ = window.__SGD__ || {}

window.__SGD__.applicationOperatorSpecificConfiguration = {
  "endpoints": {
    "scoreboard": "https://scoreboard.prod.yellow.dan.eu-central-1.sb-live.obenv.net",
    "builder": "https://sg.red7sportstech.com/bldr/static/js/main.js"
  },
  "operatorSpecificConfig": {
    "enableCasino": false,
    "folgSystemUrl": "folg-system",
    "baseUrl": "https://danskespil.dk/",
    "isOperatorSvgEnabled": true,
    "poolsUrl": "tips/",
    "sportsbookUrl": "oddset/"
  },
  "imgConfig": {
    "environment": "dev"
  },
  "google_tag_manager_id": "",
  "openbet_google_tag_manager_id": "",
  "forceMFALogin": false,
  "supportedLanguages": [
    {
      "locale": "da-DK",
      "default": true,
      "id": "da",
      "label": "Danish"
    }
  ],
  "excludeUserPreferenceKeysFromLocalStorage": [],
  "betEditorConfig": {
    "autoCashout": {
      "claimWinLimitEnabled": false,
      "stopLossLimitEnabled": false
    },
    "partialCashout": true,
    "partialCashoutInitialState": "max",
    "cashoutEnabled": true,
    "legacyPolling": {
      "pollIntervals": {
        "focused": 120000,
        "notFocused": 600000
      },
      "enable": false
    },
    "hierarchyData": {
      "enableCommentaryData": true,
      "enableEventHyperLink": true
    }
  },
  "visualisationProviders": {
    "perform": "https://secure.danskespil.performgroup.com/streaming/watch/event/index.html?partnerId=86&eventId=:externalid&userId=001&flash=y&widgetvis=true&vis3D=true&imageconfig=https://assets.sb.danskespil.dk/front-end/assets/perform-image-config.js&lang=da&width=:width",
    "spin": "https://scoreboards.sportingsolutions.com/american_football.html#event_id=:externalid&customer=danskespil&language=dk",
    "betgenius": "https://danskespil.betstream.betgenius.com/betstream-view/tennisscorecentre/danskespiltennisscorecentre/html?eventId=:externalid&culture=da-DK"
  },
  "eventPoolsNavigation": {
    "baseUrl": ""
  },
  "virtuals": {
    "operatorName": "reference",
    "inspiredMainDomain": "devqa.inspiredvss.co.uk",
    "sportsIds": []
  },
  "externalStatsProviders": {
    "enetpulse": "https://danskespil.enetscores.com/h2h/:sport/:externalid"
  },
  "mockHeaderSettings": {
    "keycloak": {
      "realm": "",
      "clientId": "",
      "url": ""
    },
    "enabled": false
  },
  "basePath": "/front-end",
  "barcode": {
    "enabled": true
  },
  "liveScoresProviders": {
    "enetscores": "https://danskespil-dk.enetscores.com/page/mx/danskespil/livescore/1"
  },
  "builder": {
    "enabled": true
  },
  "iamServiceConfig": {},
  "externalScoreboardProviders": {
    "betgenius": "https://danskespil.betstream.betgenius.com/betstream-view/multisportscoreboard/danskespilmultisportscoreboard/html?eventId=:externalid"
  },
  "betslipConfig": {
    "helpInfoBetTypes": [
      "DBL",
      "TRIXIE",
      "TBL",
      "PATENT",
      "ACCUMULATOR4",
      "ACCUMULATOR5",
      "ACCUMULATOR6",
      "ACCUMULATOR7",
      "ACCUMULATOR8",
      "ACCUMULATOR9",
      "ACCUMULATOR10",
      "ACCUMULATOR11",
      "ACCUMULATOR12",
      "ACCUMULATOR13",
      "ACCUMULATOR14",
      "ACCUMULATOR15",
      "ACCUMULATOR16",
      "ACCUMULATOR17",
      "ACCUMULATOR18",
      "ACCUMULATOR19",
      "YANKEE",
      "CANADIAN",
      "HEINZ",
      "SUPERHEINZ",
      "GOLIATH",
      "LUCKY15",
      "LUCKY31",
      "LUCKY63"
    ]
  }
}


// Environment specific configuration
// Environment specific configuration. Written inside environment-configuration.js which is injected in app-configuration.js via SSI
window.__SGD__ = window.__SGD__ || {}
window.__SGD__.environmentConfiguration =  {
  baseDomain: 'https://danskespil.dk',
  basePath: '/front-end',
  assetPath: 'https://assets.sb.danskespil.dk/front-end',
  endpoints: {
    account: 'https://api.sb.danskespil.dk/account-service/api/v1',
    bet: 'https://api.sb.danskespil.dk/bet-service/api/v1',
    identity: 'https://auth.sb.danskespil.dk/identity-service/api/v1',
    payment: 'https://api.sb.danskespil.dk/payment-service/api/v1',
    content: 'https://content.sb.danskespil.dk/content-service/api/v1',
    content_service: 'https://content.sb.danskespil.dk/content-service/api/v1',
    promotion: 'https://api.sb.danskespil.dk/promotions-service/api/v0',
    sports: 'https://api.sb.danskespil.dk/sports-service/api/v1',
    favourite: 'https://api.sb.danskespil.dk/sports-service/api/v1',
    loyalty: 'https://api.sb.danskespil.dk/loyalty-service/api/v1',
    liveserv: [
      'wss://push.sb.danskespil.dk/websock',
      'https://push.sb.danskespil.dk/push'
    ],
    bracketChallenge: 'https://api.sb.danskespil.dk/bracket-service/api/v1',
  },
  red7scripts: {
    marchMadness: '(none)://(none)/js/main.js'
  },

  // TODO - CANT_MIGRATE_ERROR. Can we move to operator webapp base, or change config format to migrate?
  virtuals: {
    sportsIds: [{"competitions": [{"id": 21, "inspired_competition_name": "soccer3"}, {"id": 703, "inspired_competition_name": "soccer3"}, {"id": 24, "inspired_competition_name": "soccer3"}, {"id": 25, "inspired_competition_name": "soccer3"}], "id": 49}, {"competitions": [{"id": 26, "inspired_competition_name": "soccer3"}, {"id": 27, "inspired_competition_name": "soccer3"}], "id": 44}, {"competitions": [{"id": 28, "inspired_competition_name": "soccer3"}], "id": 45}, {"competitions": [{"id": 704, "inspired_competition_name": "basketball"}, {"id": 705, "inspired_competition_name": "basketball"}], "id": 50}, {"competitions": [{"id": 706, "inspired_competition_name": "soccer3"}], "id": 51}]
  },
}


window.__SGD__.webapp = window.__SGD__.webapp || {};

(() => {
  const isObject = (item) => (item && typeof item === 'object' && !Array.isArray(item));

  const deepMerge = (target, ...sources) => {
    if (!sources.length) return target;
    const source = sources.shift();

    if (isObject(target) && isObject(source)) {
      for (const key in source) {
        if (isObject(source[key])) {
          if (!target[key]) {
            target[key] = {};
          }
          deepMerge(target[key], source[key]);
        } else {
          target[key] = source[key];
        }
      }
    }

    return deepMerge(target, ...sources);
  }


  window.__SGD__.webapp = deepMerge(
    window.__SGD__.applicationOperatorSpecificConfiguration,
    window.__SGD__.environmentConfiguration,
    window.__SGD__.webapp,
  );
})();

  window.__SGD__.webapp.basePath = window.__SGD__.webapp.basePath || '';
  window.__SGD__.webapp.assetPath = window.__SGD__.webapp.assetPath || '';
  
let isRetailMode = false;


if (isRetailMode) {
  window.__SGD__.webapp.featureSwitches = {
    ...(window.__SGD__.webapp.featureSwitches || {}),
    authenticationMode: 'WRAPPER_TOKEN',
    accountProvider: null,
  };
  window.__SGD__.webapp.operatorSpecificConfig = {
    ...(window.__SGD__.webapp.operatorSpecificConfig || {}),
    operatorVariation: "headless-example",
  };

  window.__SGD__.webapp.applications = {
    ssbt: {
      engineCustomisation: {
        config: {
          playerSessionConfigs: null
        }
      }
    }
  };
}

  
  console.warn('digital-portal.js did not specify any CSS files')
  window.dispatchEvent(new CustomEvent('@spa/css-loaded'));
  

  const jsBundleFiles = ["/assets/runtime.5f560c03c43e843b02d3.bundle.js","/assets/2899.0786a5f99ea5f5bb865e.bundle.js","/assets/9151.daab463a547768640c39.bundle.js","/assets/6008.019353fe613aa19b8485.bundle.js","/assets/8428.d422dd0507c49c2a1a0c.bundle.js","/assets/4401.db8e2ed580fc9fa7dc33.bundle.js","/assets/8879.2ae035417b110ad062aa.bundle.js","/assets/3528.f653d4aa428764b2157a.bundle.js","/assets/3786.1ce8db94096faa6f3232.bundle.js","/assets/7552.e559cb6d00cf0e7774c9.bundle.js","/assets/3475.dbce2632a66972f84551.bundle.js","/assets/1934.ee8a10c1d9071741478f.bundle.js","/assets/8284.0338f9c43cdf087436b4.bundle.js","/assets/4943.2bfeddee8b2a4367e579.bundle.js","/assets/1001.18565045ce5a1be12217.bundle.js","/assets/311.0fd99bb68300dd9dba4f.bundle.js","/assets/713.f2442db3cf835f85c6b5.bundle.js","/assets/9818.a169c5814092022cb36a.bundle.js","/assets/4161.02626fdeec31deafd1ec.bundle.js","/assets/6621.b91558fd17cf66c8309b.bundle.js","/assets/5671.af1ca1ed1411808390ea.bundle.js","/assets/997.e6d232759d98856f16e8.bundle.js","/assets/568.872c7af1272eb27a0e98.bundle.js","/assets/4459.9fdce4a1201486286f5f.bundle.js","/assets/2972.864d44c232226dfb3909.bundle.js","/assets/7019.8f1b487f0aa08984a80a.bundle.js","/assets/8532.a30b9aced599767dc899.bundle.js","/assets/5933.e88c07da0040b9c8818e.bundle.js","/assets/8821.fec95279432e2f01653f.bundle.js","/assets/2478.2265bf35a765fabbd14e.bundle.js","/assets/2624.d0d650835a5b12ae7243.bundle.js","/assets/9489.cb8dda7ff94197ec367a.bundle.js","/assets/9038.bd6c7b1ecfc17c381ea6.bundle.js","/assets/9521.68faf7c8a6649f384b57.bundle.js","/assets/6661.2cbc40cc12d7a50c8169.bundle.js","/assets/1679.3cef4306f45c92a0cd4e.bundle.js","/assets/59.2ee0b3b0132c3bfecc94.bundle.js","/assets/6970.98feab9a96fe03be1be9.bundle.js","/assets/2010.4c0a0da12993141f7642.bundle.js","/assets/204.7cffe5ea35af6315f21f.bundle.js","/assets/8782.fcc9b027ec588e0e4d49.bundle.js","/assets/7817.95947e6b8163ffb76928.bundle.js","/assets/3038.26a5c4c231dde1a76be8.bundle.js","/assets/78.4de2b159e6399295d22b.bundle.js","/assets/8454.bdfb047f02f7ccd3eafb.bundle.js","/assets/1703.dccdddea2aa55bcf9b58.bundle.js","/assets/2529.7d3005ec15cc797a0282.bundle.js","/assets/7297.0a58a588c332f662e70e.bundle.js","/assets/3825.c243b87aa165fd5d5ac8.bundle.js","/assets/6486.36fabf61f9d0886e7783.bundle.js","/assets/1522.36360cabb8f6eeaf0c39.bundle.js","/assets/8487.14749e985c5f3ff3df59.bundle.js","/assets/2600.712e2b85b5421b6de207.bundle.js","/assets/main.e92a6320bc551d711f83.bundle.js"];
  jsBundleFiles.forEach(src => {
    const scriptTag = document.createElement('script');
    scriptTag.src = window.__SGD__.webapp.assetPath + src + '?v=49.4.31-SNAPSHOT';
    scriptTag.async = true;
    document.body.appendChild(scriptTag);
  });

})(window)