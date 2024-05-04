import logo from './logo.svg';
import './App.css';

function App() {
  const handleFileUpload = async (event) => {
    event.preventDefault();

    if (event.target.files.length === 0) {
      console.error('No file selected');
      return;
    }

    const formData = new FormData();
    formData.append('file', event.target.files[0]);

    const queryString = new URLSearchParams(formData).toString();

    const url = `http://localhost:8000/getRelatedGarments?${queryString}`;

    try {
      const response = await fetch(url).then((response) => response.json().then((data) => console.log(data)));

    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    // Get an image from the user
    <div className="App">
      <header className="App-header">
        <form>
          <input type="file" onChange={handleFileUpload} />
          <button type="submit">Upload</button>
        </form>
      </header>
    </div>
  );
}

export default App;
