// import logo from './logo.svg';
import './App.css';
import { ChangeEvent, useState } from 'react';
import FileUploadSingle from "./FileUploadSingle";

function sayHello() {
  alert('This feature is under development!');
}



function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>
          Connectome project
        </h1>
          {/*<button onClick={sayHello}><h3>Upload patient image</h3></button>*/}
        <FileUploadSingle/>
      </header>
    </div>

  );
}

export default App;
