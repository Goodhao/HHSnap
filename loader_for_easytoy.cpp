#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <windows.h>
#define FOR(i,n) for (int i=1;i<=n;i++)
#define REP(i,a,b) for (int i=a;i<=b;i++)
using namespace std;
using namespace cv;

const int N = 5000 + 10;
int h, w;
vector<int> color_junction = { 255,0,0 };
vector<int> color_stroke = { 0,255,0 };
bool in(int x, int y) {
    if (0 <= x && x < h && 0 <= y && y < w) return 1;
    else return 0;
}
int next_odd(int x) {
    if (x % 2) return x;
    else return x + 1;
}
void remove_background(vector<vector<int>> img) {
    queue<pair<int, int>> q;
    q.push(make_pair(0, 0)); // 确保左上角一定是背景点，从它开始遍历所有背景点
    while (q.size()) {
        auto t = q.front();
        int x = t.first, y = t.second;
        q.pop();
        REP(i, -1, 1) REP(j, -1, 1) {
            if (i == 0 || j == 0) {
                if (!(i == 0 && j == 0)) {
                    if (in(x + i, y + j) && img[x + i][y + j] == 255) {
                        img[x + i][y + j] = 0;
                        q.push(make_pair(x + i, y + j));
                    }
                }
            }
        }
    }
}
vector<vector<int>> min_filter(const vector<vector<int>> &img) {
    vector<vector<int>> res(h, vector<int>(w, 255));
    REP(x, 0, h - 1) REP(y, 0, w - 1) {
        REP(i, -1, 1) REP(j, -1, 1) {
            if (in(x + i, y + j)) {
                res[x][y] = min(res[x][y], img[x + i][y + j]);
            }
        }
    }
    return res;
}
vector<vector<int>> linear_dodge(const vector<vector<int>> &img1, const vector<vector<int>> &img2) {
    vector<vector<int>> res(h, vector<int>(w, 255));
    REP(x, 0, h - 1) REP(y, 0, w - 1) {
        res[x][y] = min(255, img1[x][y] + img2[x][y]);
    }
    return res;
}
vector<vector<int>> rotate(vector<vector<int>>& a) {
    vector<vector<int>> res(3, vector<int>(3, 0));
    REP(i, 0, 2) REP(j, 0, 2) res[2 - j][i] = a[i][j];
    return res;
}
set<pair<int, int>> process(vector<vector<int>>& img, vector<pair<int, int>> pos, vector<vector<vector<int>>> e, bool in_place = false) {
    set<pair<int, int>> res;
    int x, y;
    for (auto p : pos) {
        x = p.first, y = p.second;
        for (int k = 0; k < e.size(); k++) {
            bool drop = true;
            REP(i, -1, 1) REP(j, -1, 1) {
                int val = (in(x + i, y + j) ? img[x + i][y + j] : 0);
                if (e[k][1 + i][1 + j] == 1 && val == 0) drop = false;
                if (e[k][1 + i][1 + j] == 0 && val != 0) drop = false;
            }
            if (in_place) {
                if (drop) {
                    img[x][y] = 0;
                    break;
                }
            }
            else {
                if (drop) {
                    res.insert(make_pair(x, y));
                    break;
                }
            }
        }
    }
    return res;
}
bool prune(vector<vector<int>>& img, const vector<pair<int, int>> &pos, const set<pair<int, int>> &endpoint, const set<pair<int, int>> &junction) {
    bool update = false;
    int x, y;
    for (auto p : pos) {
        x = p.first, y = p.second;
        if (img[x][y] == 0) continue;
        bool alone = true;
        REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && in(x + i, y + j) && img[x + i][y + j] != 0) {
            alone = false;
            break;
        }
        if (alone) {
            img[x][y] = 0;
            update = true;
        }
    }
    for (auto p : endpoint) {
        x = p.first, y = p.second;
        if (img[x][y] != 0) {
            queue<pair<int, int>> q;
            q.push(make_pair(x, y));
            img[x][y] = 0;
            update = true;
            while (q.size()) {
                auto t = q.front();
                q.pop();
                x = t.first, y = t.second;
                bool stop = false;
                REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && in(x + i, y + j) && img[x + i][y + j] != 0 && junction.find(make_pair(x + i, y + j)) != junction.end()) {
                    stop = true;
                }
                if (stop) continue;
                REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && in(x + i, y + j) && img[x + i][y + j] != 0) {
                    img[x + i][y + j] = 0;
                    q.push(make_pair(x + i, y + j));
                }
            }
        }
    }
    return update;
}
void pick_stroke(vector<pair<int, int>>& res, vector<vector<vector<int>>> &img, int x, int y) {
    res.push_back(make_pair(x, y));
    img[x][y] = vector<int>{ 0,0,0 };
    REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && in(x + i, y + j) && img[x + i][y + j] == color_stroke) {
        pick_stroke(res, img, x + i, y + j);
        break;
    }
}
pair<vector<vector<pair<int, int>>>, map<pair<int,int>, int>> build_sketch(vector<vector<vector<int>>> img) {
    vector<vector<pair<int, int>>> sketch;
    map<pair<int, int>, int> which_stroke;
    REP(x, 0, h - 1) REP(y, 0, w - 1) if (img[x][y] == color_stroke) {
        int deg = 0;
        REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && in(x + i, y + j) && img[x + i][y + j] == color_stroke) deg++;
        if (deg == 1) {
            vector<pair<int, int>> stroke;
            pick_stroke(stroke, img, x, y);
            for (auto &p : stroke) which_stroke[p] = sketch.size();
            sketch.push_back(stroke);
        }
    }
    // 剩下的是成环的
    REP(x, 0, h - 1) REP(y, 0, w - 1) if (img[x][y] == color_stroke) {
        vector<pair<int, int>> stroke;
        pick_stroke(stroke, img, x, y);
        for (auto &p : stroke) which_stroke[p] = sketch.size();
        sketch.push_back(stroke);
    }
    return make_pair(sketch, which_stroke);
}
vector<vector<int>> tmp;
vector<pair<int,int>> flood(set<int>& res, const vector<vector<vector<int>>>& img, const map<pair<int, int>, int>& which_stroke, vector<vector<int>>& region, int x, int y, int region_ID = 1) {
    queue<pair<int, int>> q;
    vector<pair<int, int>> change;
    q.push(make_pair(x, y));
    change.push_back(make_pair(x, y));
    region[x][y] = region_ID;
    int xmin = x, xmax = x, ymin = y, ymax = y;
    while (q.size()) {
        auto t = q.front();
        q.pop();
        x = t.first, y = t.second;
        xmin = min(xmin, x);
        xmax = max(xmax, x);
        ymin = min(ymin, y);
        ymax = max(ymax, y);
        REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && (i == 0 || j == 0) && in(x + i, y + j)) if (region[x + i][y + j] == 0) {
            region[x + i][y + j] = region_ID;
            q.push(make_pair(x + i, y + j));
            change.push_back(make_pair(x + i, y + j));
        }
    }
    xmin = max(0, xmin - 1);
    ymin = max(0, ymin - 1);
    xmax = min(h - 1, xmax + 1);
    ymax = min(w - 1, ymax + 1);
    if (region[0][0] == region_ID) {
        // 背景区域，确保左上角是背景点
        REP(x, 0, h - 1) REP(y, 0, w - 1) if (region[x][y] == region_ID) {
            REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && (i == 0 || j == 0) && in(x + i, y + j)) {
                if (region[x + i][y + j] != region_ID && (img[x + i][y + j] == color_stroke)) {
                    res.insert(which_stroke.at(make_pair(x + i, y + j)));
                }
            }
        }
        return change;
    }
    assert(q.empty());
    pair<int, int> s;
    REP(x, xmin, xmax) REP(y, ymin, ymax) if (region[x][y] != region_ID) {
        s = make_pair(x, y);
    }
    tmp[s.first][s.second] = region_ID;
    q.push(s);
    while (q.size()) {
        auto t = q.front();
        q.pop();
        x = t.first, y = t.second;
        REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && (i == 0 || j == 0) && xmin <= x && x <= xmax && ymin <= y && y <= ymax) {
            if (region[x + i][y + j] != region_ID && tmp[x + i][y + j] != region_ID) {
                tmp[x + i][y + j] = region_ID;
                q.push(make_pair(x + i, y + j));
            }
        }
    }
    REP(x, xmin, xmax) REP(y, ymin, ymax) if (region[x][y] == region_ID) {
        REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && (i == 0 || j == 0) && xmin <= x && x <= xmax && ymin <= y && y <= ymax) {
            if (tmp[x + i][y + j] == region_ID && img[x + i][y + j] == color_stroke) {
                res.insert(which_stroke.at(make_pair(x + i, y + j)));
            }
        }
    }
    /*
    REP(x, xmin, xmax) REP(y, ymin, ymax) if (tmp[x][y] != region_ID) {
        REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && (i == 0 || j == 0) && xmin <= x && x <= xmax && ymin <= y && y <= ymax) {
            if (tmp[x + i][y + j] == region_ID && img[x + i][y + j] == color_stroke) {
                res.insert(which_stroke.at(make_pair(x + i, y + j)));
            }
        }
    }*/
    return change;
}
vector<set<int>> build_region(const vector<vector<vector<int>>>& img, const map<pair<int, int>, int>& which_stroke) {
    vector<vector<int>> region(h, vector<int>(w, 0));
    tmp = vector<vector<int>>(h, vector<int>(w, 0));
    REP(x, 0, h - 1) REP(y, 0, w - 1) if (img[x][y] != vector<int>{0, 0, 0}) {
        region[x][y] = -1;
    }
    vector<set<int>> stroke_sets;
    int region_ID = 1;
    REP(x, 0, h - 1) REP(y, 0, w - 1) {
        if (region[x][y] == 0) {
            set<int> s;
            auto change = flood(s, img, which_stroke, region, x, y, region_ID);
            if (s.size() > 0) {
                if (!(x == 0 && y == 0)) stroke_sets.push_back(s);
                region_ID++;
            }
            else {
                for (auto p : change) {
                    region[p.first][p.second] = 0;
                }
            }
        }
    }
    return stroke_sets;
}
int buffer[1024 * 1024];
int main() {

    HANDLE hPipe;
    DWORD dwRead;
    

    // 连接命名管道
    cout << "Connecting to named pipe..." << endl;
    hPipe = CreateFile(TEXT("\\\\.\\pipe\\MyPipe2"), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (hPipe == INVALID_HANDLE_VALUE)
    {
        cout << "Failed to connect to named pipe" << endl;
        return 1;
    }
    cout << "Connected" << endl;

    ReadFile(hPipe, buffer, 2 * sizeof(int), &dwRead, NULL);
    int H, W;
    H = buffer[0];
    W = buffer[1];
    h = H;
    w = W;

    ReadFile(hPipe, buffer, H * W * 3 * sizeof(int), &dwRead, NULL);
    Mat img(H, W, CV_8UC3, Scalar(0, 0, 0));
    REP(i, 0, H - 1) REP(j, 0, W - 1) REP(k, 0, 2) {
        int c = buffer[i * W * 3 +j * 3 + k];
        img.at<Vec3b>(i, j)[k] = c;
    }

    Mat gray, binary;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    vector<vector<int>> a(h, vector<int>(w, 255));
    REP(i, 0, h - 1) REP(j, 0, w - 1) a[i][j] = gray.at<unsigned char>(i, j);
    auto b = min_filter(a);
    a = linear_dodge(a, b);
    REP(i, 0, h - 1) REP(j, 0, w - 1) gray.at<unsigned char>(i, j) = a[i][j];
    threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    REP(i, 0, h - 1) REP(j, 0, w - 1) {
        unsigned char& c = binary.at<unsigned char>(i, j);
        if (c == 255) c = 0;
        else if (c == 0) c = 255;
    }
    // thining
    REP(i, 0, h - 1) REP(j, 0, w - 1) {
        if (binary.at<unsigned char>(i, j) == 255) a[i][j] = 1;
        else a[i][j] = 0;
    }
    int cnt = 0;
    while (1) {
        bool flag = 1;
        vector<vector<int>> mask(h, vector<int>(w, 0));
        REP(x, 0, h - 1) REP(y, 0, w - 1) {
            if (a[x][y] == 0) continue;
            vector<vector<int>> P(3, vector<int>(3, 0));
            REP(i, -1, 1) REP(j, -1, 1) {
                if (in(x + i, y + j)) P[1 + i][1 + j] = a[x + i][y + j];
                else P[1 + i][1 + j] = 0;
            }
            int sum = 0;
            REP(i, 0, 2) REP(j, 0, 2) {
                P[i][j] = 1 - P[i][j];
                sum += P[i][j];
            }
            int B = sum - P[1][1];
            vector<int> t = { P[0][1],P[0][2],P[1][2],P[2][2],P[2][1],P[2][0],P[1][0],P[0][0],P[0][1] };
            int A = 0;
            REP(i, 0, t.size() - 2) {
                if (t[i] == 0 && t[i + 1] == 1) {
                    A++;
                }
            }
            if (2 <= B && B <= 6 && A == 1) {
                if (cnt % 2 == 0) {
                    if (P[0][1] + P[1][2] + P[2][1] >= 1 && P[1][2] + P[2][1] + P[1][0] >= 1) {
                        mask[x][y] = 1;
                        flag = 0;
                    }
                }
                else {
                    if (P[0][1] + P[1][2] + P[1][0] >= 1 && P[0][1] + P[2][1] + P[1][0] >= 1) {
                        mask[x][y] = 1;
                        flag = 0;
                    }
                }
            }
        }
        cnt++;
        REP(x, 0, h - 1) REP(y, 0, w - 1) a[x][y] ^= mask[x][y];
        if (flag) break;
    }
    REP(i, 0, h - 1) REP(j, 0, w - 1) {
        if (a[i][j] == 1) binary.at<unsigned char>(i, j) = 255;
        else binary.at<unsigned char>(i, j) = 0;
    }
    cout << "thining ok" << endl;

    // pruning
    vector<vector<vector<int>>> eight_connected(4, vector<vector<int>>(3, vector<int>(3, 0)));
    eight_connected[0] = vector<vector<int>>{ vector<int>{-1, 1, -1} , vector<int>{-1, 1, 1}, vector<int>{0, -1, -1} };
    REP(i, 1, 3) eight_connected[i] = rotate(eight_connected[i - 1]);
    vector<vector<vector<int>>> compute_junction(12, vector<vector<int>>(3, vector<int>(3, 0)));
    compute_junction[0] = vector<vector<int>>{ vector<int>{-1, 1, -1} , vector<int>{0, 1, 0}, vector<int>{1, 0, 1} };
    REP(i, 1, 3) compute_junction[i] = rotate(compute_junction[i - 1]);
    compute_junction[4] = vector<vector<int>>{ vector<int>{1, 0, -1} , vector<int>{0, 1, 0}, vector<int>{1, 0, 1} };
    REP(i, 5, 7) compute_junction[i] = rotate(compute_junction[i - 1]);
    compute_junction[8] = vector<vector<int>>{ vector<int>{-1, 0, 1} , vector<int>{1, 1, 0}, vector<int>{-1, 1, -1} };
    REP(i, 9, 11) compute_junction[i] = rotate(compute_junction[i - 1]);
    vector<vector<vector<int>>> compute_endpoint(8, vector<vector<int>>(3, vector<int>(3, 0)));
    compute_endpoint[0] = vector<vector<int>>{ vector<int>{-1, 1, -1} , vector<int>{0, 1, 0}, vector<int>{0, 0, 0} };
    REP(i, 1, 3) compute_endpoint[i] = rotate(compute_endpoint[i - 1]);
    compute_endpoint[4] = vector<vector<int>>{ vector<int>{1, 0, 0} , vector<int>{0, 1, 0}, vector<int>{0, 0, 0} };
    REP(i, 5, 7) compute_endpoint[i] = rotate(compute_endpoint[i - 1]);

    vector<pair<int, int>> pos;
    REP(i, 0, h - 1) REP(j, 0, w - 1) {
        a[i][j] = binary.at<unsigned char>(i, j);
    }
    REP(x, 0, h - 1) REP(y, 0, w - 1) {
        if (a[x][y] == 255) pos.push_back(make_pair(x, y));
    }
    process(a, pos, eight_connected, true);
    auto junction = process(a, pos, compute_junction);
    auto endpoint = process(a, pos, compute_endpoint);
    while (prune(a, pos, endpoint, junction)) {
        junction = process(a, pos, compute_junction);
        endpoint = process(a, pos, compute_endpoint);
    }
    cout << "pruning ok" << endl;

   
    vector<vector<vector<int>>> out(h, vector<vector<int>>(w, vector<int>(3, 0)));
    REP(x, 0, h - 1) REP(y, 0, w - 1) {
        if (junction.find(make_pair(x, y)) != junction.end()) {
            out[x][y] = color_junction;
        }
        else if (a[x][y] != 0) {
            out[x][y] = color_stroke;
        }
        else {
            out[x][y] = vector<int>{ 0,0,0 };
        }
    }
    for (auto p : junction) {
        int x = p.first, y = p.second;
        REP(i, -1, 1) REP(j, -1, 1) if (!(i == 0 && j == 0) && in(x + i, y + j) && out[x + i][y + j] == color_stroke) {
            out[x + i][y + j] = color_junction;
        }
    }
    REP(i, 0, h - 1) REP(j, 0, w - 1) REP(k, 0, 2) {
        img.at<Vec3b>(i, j)[k] = out[i][j][k];
    }
    // imshow("out", img);
    // waitKey(0);
    auto p = build_sketch(out);
    auto sketch = p.first;

    int cur = 0;
    buffer[cur++] = sketch.size();
    for (int i = 0; i < sketch.size(); i++) {
        buffer[cur++] = sketch[i].size();
        for (int j = 0; j < sketch[i].size(); j++) {
            buffer[cur++] = sketch[i][j].first;
            buffer[cur++] = sketch[i][j].second;
        }
    }
    WriteFile(hPipe, buffer, cur * sizeof(int), &dwRead, NULL);
    cout << "finish" << endl;
    imwrite("out_easytoy.png", img);
    //auto which_stroke = p.second;
    //auto stroke_sets = build_region(out, which_stroke);
    //for (auto s : stroke_sets) {
    //    cout << "{";
    //    for (auto x : s) {
    //        if (x == *s.begin()) cout << x;
    //        else cout << "," << x;
    //    }
    //    cout << "}" << endl;
    //}
    //// imwrite("out_easytoy.png", img);

    //ofstream fout;

    //fout.open("sketch.txt");
    //fout << sketch.size() << endl;
    //for (auto stroke : sketch) {
    //    for (auto p : stroke) fout << p.first << " " << p.second << " ";
    //    fout << endl;
    //}
    //fout.close();

    //fout.open("stroke_sets.txt");
    //fout << stroke_sets.size() << endl;
    //for (auto s : stroke_sets) {
    //    for (auto x : s) fout << x << " ";
    //    fout << endl;
    //}
    //fout.close();
 
    //fout.open("which_stroke.txt");
    //fout << which_stroke.size() << endl;
    //for (auto p : which_stroke) {
    //    fout << p.first.first << " " << p.first.second << " " << p.second << endl;
    //}
    //fout.close();
    return 0;
}