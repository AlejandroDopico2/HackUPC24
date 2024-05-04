import logo from './logo.svg';
import './App.css';

function App() {
  return (
    // Get an image from the user
    <div className="App">
      <header className="App-header">
        <form>
          <input type="file" />
          <button>Upload</button>
        </form>
      </header>
    </div>
  );
}

export default App;
