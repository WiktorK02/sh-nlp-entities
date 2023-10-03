# sh-nlp-entities
Smart Home NLP entities recognition 
learned entities:
light, kitchen, garage, (any)number<br>

text:
```
text = "One light in the kitchen."
```
return:
```
{'One': 'NUMBER', 'light': 'DEVICE', 'kitchen': 'PLACE'}
```
## Task list
* learn other devices, places ect
* return as json
* conncect number of devices with devicces and places
## How to Contribute
1. Fork the Project
2. Clone repo with your GitHub username instead of ```YOUR-USERNAME```:<br>
```
$ git clone https://github.com/YOUR-USERNAME/sh-nlp-entities
```
3. Create new branch:<br>
```
$ git branch BRANCH-NAME 
$ git checkout BRANCH-NAME
```
4. Make changes and test<br>
5. Submit Pull Request with comprehensive description of change
## License 
[MIT license](LICENSE)
