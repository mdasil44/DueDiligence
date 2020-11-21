import './App.css';
import React from 'react';
import CustomizedSlider from './CustomizedSlider.js';
import Typography from '@material-ui/core/Typography';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Due Diligence</h1>
        <NameForm />
        <p id="Selectors">
          <Typography gutterBottom>Desired Confidence Level</Typography>
          <CustomizedSlider />
        </p>
        <p id="Recommend">
          <Typography gutterBottom>Suggestion</Typography>
          <CustomizedSlider />
        </p>
      </header>
    </div>
  );
}

export class NameForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = { value: '' };

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event) {
    this.setState({ value: event.target.value });
  }

  handleSubmit(event) {
    alert('A name was submitted: ' + this.state.value);
    event.preventDefault();
  }

  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <label for="Stock-input">
          Enter Stock:
          &nbsp;
          <input type="text" id="Stock-input" value={this.state.value} onChange={this.handleChange} />
        </label>
        <input type="submit" id="Stock-input" value="Submit" />
      </form>
    )
  }
}

export default App;
