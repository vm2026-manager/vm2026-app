from pathlib import Path
from datetime import datetime

path = Path("index.html")
text = path.read_text(encoding="utf-8")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = Path(f"index.backup_before_fix_fixture_tooltip_lookup_{timestamp}.html")
backup.write_text(text, encoding="utf-8")

old = '''    async function fixturesFor(player) {
      var odds = await getOdds();
      var team = String(player.team_id || "").toUpperCase();
      var out = [];

      [1, 2, 3].forEach(function (roundNo) {
        var opponent = String(
          player["match_" + roundNo + "_opponent_team"] ||
          player["round" + roundNo + "_opponent_team"] ||
          ""
        ).toUpperCase();

        if (!opponent || opponent === "0") return;

        var row = odds.get(team + "|" + opponent);
        var kickoff = String(
          player["match_" + roundNo + "_kickoff_dk"] ||
          player["round" + roundNo + "_kickoff_dk"] ||
          (row ? row.kickoff : "") ||
          ""
        );

        if (!isFuture(kickoff)) return;

        var win = null;
        var draw = null;
        var loss = null;

        if (row) {
          if (row.home === team) {
            win = row.h;
            draw = row.x;
            loss = row.a;
          } else if (row.away === team) {
            win = row.a;
            draw = row.x;
            loss = row.h;
          }
        }

        out.push({
          opponent: opponent,
          opponentName: teamName(opponent),
          kickoff: kickoff,
          win: win,
          draw: draw,
          loss: loss,
          source: row ? row.source : ""
        });
      });

      return out.slice(0, 3);
    }'''

new = '''    async function fixturesFor(player) {
      var odds = await getOdds();
      var team = canonicalTeamId(player.team_id || player.team_name || "");
      var out = [];

      function addFixture(home, away, kickoff, matchId, explicitOpponent) {
        home = canonicalTeamId(home);
        away = canonicalTeamId(away);
        var opponent = canonicalTeamId(explicitOpponent || (home === team ? away : home));

        if (!team || !opponent || opponent === "0") return;
        if (!kickoff) {
          var kickoffRow = odds.get(home + "|" + away) || odds.get(away + "|" + home);
          kickoff = kickoffRow ? kickoffRow.kickoff : "";
        }
        if (!isFuture(kickoff)) return;

        var row = odds.get(home + "|" + away) || odds.get(away + "|" + home);
        if (!row && matchId) {
          row = matchOddsRows.find(function (candidate) {
            return String(getRowValue(candidate, ["match_id", "fixture_id", "id"])) === String(matchId);
          }) || null;
        }

        var win = null;
        var draw = null;
        var loss = null;

        if (row) {
          var rowHome = canonicalTeamId(row.home);
          var rowAway = canonicalTeamId(row.away);

          if (rowHome === team) {
            win = row.h;
            draw = row.x;
            loss = row.a;
          } else if (rowAway === team) {
            win = row.a;
            draw = row.x;
            loss = row.h;
          }
        }

        out.push({
          opponent: opponent,
          opponentName: teamName(opponent),
          kickoff: kickoff,
          win: win,
          draw: draw,
          loss: loss,
          source: row ? row.source : ""
        });
      }

      [1, 2, 3].forEach(function (roundNo) {
        var opponent = player["match_" + roundNo + "_opponent_team"] ||
          player["round" + roundNo + "_opponent_team"] ||
          "";

        if (!opponent || String(opponent) === "0") return;

        var home = player["match_" + roundNo + "_home"] ||
          player["round" + roundNo + "_home"] ||
          team;

        var away = player["match_" + roundNo + "_away"] ||
          player["round" + roundNo + "_away"] ||
          opponent;

        var kickoff = String(
          player["match_" + roundNo + "_kickoff_dk"] ||
          player["round" + roundNo + "_kickoff_dk"] ||
          ""
        );

        addFixture(home, away, kickoff, player["match_" + roundNo + "_id"] || player["round" + roundNo + "_id"], opponent);
      });

      if (!out.length && Array.isArray(fixtures)) {
        fixtures
          .filter(function (f) { return f.stage === "GROUP"; })
          .filter(function (f) {
            return canonicalTeamId(f.home) === team || canonicalTeamId(f.away) === team;
          })
          .filter(function (f) {
            return f.kickoffDate instanceof Date &&
              !Number.isNaN(f.kickoffDate.getTime()) &&
              f.kickoffDate >= getNow();
          })
          .sort(function (a, b) { return a.kickoffDate - b.kickoffDate; })
          .slice(0, 3)
          .forEach(function (f) {
            addFixture(f.home, f.away, f.kickoff_dk || f.kickoff || "", f.match_id);
          });
      }

      return out.slice(0, 3);
    }'''

if old not in text:
    raise SystemExit("Kunne ikke finde den eksisterende fixturesFor(player)-blok. Stopper uden ændring.")

text = text.replace(old, new)
path.write_text(text, encoding="utf-8")

print("Backup:", backup)
print("Rettet: tooltip fixturesFor bruger nu fixtures-fallback, hvis spilleren mangler match_1/2/3_opponent_team.")
print("Test lokalt: hover Arya Yousefi. Tooltip bør vise Iran-New Zealand/BEL/EGY-kampe i stedet for 'Ingen kommende kampe'.")
