## Compile Javascript

```bash
cd assets/reconstruction/robotic-top-down
npm install
./node_modules/bin/webpack.js ./main.js --output-filename=bundle.js --mode=production

cd assets/reconstruction/table-top-side
npm install
./node_modules/bin/webpack.js ./main.js --output-filename=bundle.js --mode=production
```
