system DateApp {

  /* 1. Page Definition: Document layout and Global Styles */
  page index {
    head:
      style body {
        font-family: sans-serif;
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f5f5f5;
      }
      style .container {
        background: white;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }
      style h1 {
        color: #333;
        border-bottom: 2px solid #324fff;
        padding-bottom: 10px;
      }
      style .instructions {
        background: #e8f4ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 4px solid #324fff;
      }
      style .history-log {
        margin-top: 30px;
        padding: 15px;
        background: #f0f0f0;
        border-radius: 8px;
        font-size: 14px;
      }
      style .history-log h3 {
        margin-top: 0;
        color: #666;
      }
      style .log-entry {
        padding: 5px 0;
        border-bottom: 1px solid #ddd;
        font-family: monospace;
      }

    body:
      DateAppRoot
  }

  /* 2. Root Layout Component */
  component DateAppRoot {
    template:
      div .container {
        h1 { "📅 DateApp" }
        div .instructions {
          p { "이 앱은 DSL(DateApp)을 HTML/CSS/JavaScript로 구현한 예제입니다." }
          p { "날짜를 선택하거나 \"Choose Today\" 버튼을 클릭하면 DateDisplay 컴포넌트가 애니메이션과 함께 업데이트됩니다." }
        }
        
        MyElement

        div .history-log {
          h3 { "📝 상태 변경 로그" }
          div #log-entries { }
        }
      }

    /* System initialization after mounting to the DOM */
    on mount:
      exposeToGlobal(this)
  }

  /* 3. Main Logic Component: State and Event Handling */
  component MyElement {
    state date : Date

    template:
      div .date-picker {
        h2 { "MyElement 컴포넌트" }
        
        div .input-group {
          p { strong { "Choose a date:" } }
          input type="date" #date-input on change -> dateChanged
          button #today-btn on click -> chooseToday { "Choose Today" }
        }

        div .status {
          p { strong { "Date chosen:" } }
        }

        div #date-display-container .date-display-container {
          DateDisplay bind date
        }
      }

    /* Transition: Event -> State Mutation */
    transition dateChanged(e):
      let utc = e.target.valueAsDate
      if utc != null:
        date = localDateFromUTC(utc)
        addLog("날짜 변경됨: " + date.toLocaleDateString())

    transition chooseToday:
      date = now()
      addLog("오늘 날짜 선택: " + date.toLocaleDateString())
  }

  /* 4. Display Component: Invariant and Visual Effects */
  component DateDisplay {
    state date : Date

    /* Invariant: Domain-level equality check for re-rendering */
    invariant:
      renderOnlyIf isSameDate(old.date, date) == false

    template:
      span #date-field .date-field {
        text date ? date.toLocaleDateString() : "날짜를 선택하세요"
      }

    /* Reactive Effect: State Change -> Imperative DOM Animation */
    on update(date):
      animate #date-field frames [
        { bg: "#fff" }, 
        { bg: "#324fff" }, 
        { bg: "#fff" }
      ] duration 1000
      addLog("DateDisplay 애니메이션 실행됨")
  }

  /* 5. Utils: Pure functions and Shared Logic */
  utils {
    function isSameDate(d1, d2) {
      if (!d1 || !d2) return false;
      return d1.toLocaleDateString() == d2.toLocaleDateString();
    }

    function localDateFromUTC(utc) {
      return new Date(utc.getUTCFullYear(), utc.getUTCMonth(), utc.getUTCDate());
    }

    function now() {
      return new Date();
    }

    /* Helper for logging state changes and updating the UI log */
    function addLog(message) {
      const container = document.getElementById("log-entries");
      if (!container) return;
      
      const entry = document.createElement("div");
      entry.className = "log-entry";
      const timestamp = new Date().toLocaleTimeString();
      entry.textContent = "[" + timestamp + "] " + message;
      
      if (container.firstChild) {
        container.insertBefore(entry, container.firstChild);
      } else {
        container.appendChild(entry);
      }
      
      // Limit logs
      while (container.children.length > 10) {
        container.removeChild(container.lastChild);
      }
    }

    /* System Export logic for DevTools exposure */
    function exposeToGlobal(instance) {
      window.DateApp = {
        utils: this,
        instance: instance,
        getCurrentDate: () => instance.date
      };
      addLog("DateApp 초기화 완료 (window.DateApp 노출)");
    }
  }
}