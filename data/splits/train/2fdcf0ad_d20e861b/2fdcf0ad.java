import java.util.*;
import java.io.*;

import static java.lang.Math.*;

public class Main {
    static Scanner sc = new Scanner(System.in);
    static PrintWriter pw = new PrintWriter(System.out);

    static int[][] map;
    static int n, m;

    static ArrayList<String> actions = new ArrayList<>();
    static String[] move = new String[]{"", "0110", "1001", "1101", "1001", "1110", "1011", "0000", "0110", "0111", "1101", "0000", "0111", "0000", "0000", "0001"};
    static int[] xdif = new int[]{0, 0, 1, 1}, ydif = new int[]{0, 1, 0, 1};

    public static void main(String[] args) throws IOException {
        int tests = sc.nextInt();
        for (int t = 0; t < tests; t++)
            solve();
        pw.close();
    }

    static void solve() throws IOException {
        n = sc.nextInt();
        m = sc.nextInt();
        map = new int[n][m];
        actions.clear();
        for (int i = 0; i < n; i++) {
            String line = sc.next();
            for (int j = 0; j < m; j++)
                map[i][j] = line.charAt(j) - '0';
        }
        if (n % 2 + m % 2 == 2)
            removeCorner();
        if (n % 2 == 1)
            for (int y = 0; y + 1 < m; y += 2)
                removeDown(y);
        if (m % 2 == 1)
            for (int x = 0; x + 1 < n; x += 2)
                removeRight(x);
        for (int x = 0; x + 1 < n; x += 2)
            for (int y = 0; y + 1 < m; y += 2)
                square(x, y);
        pw.println(actions.size());
        for (String action : actions)
            pw.println(action);
    }

    static void square(int x, int y) {
        String type = "" + map[x][y] + map[x][y + 1] + map[x + 1][y] + map[x + 1][y + 1];
        if (map[x][y] + map[x + 1][y] + map[x][y + 1] + map[x + 1][y + 1] > 0) {
            int[][] poses = new int[3][];
            int index = 0;
            for (int t = 0; t < 4; t++) {
                if (map[x + xdif[t]][y + ydif[t]] != move[Integer.parseInt(type, 2)].charAt(t) - '0')
                    poses[index++] = new int[]{x + xdif[t], y + ydif[t]};
            }
            action(poses[0][0], poses[0][1], poses[1][0], poses[1][1], poses[2][0], poses[2][1]);
            square(x, y);
        }
    }

    static void invert(int x, int y) {
        map[x][y] = 1 - map[x][y];
    }

    static void removeCorner() {
        if (map[n - 1][m - 1] == 1)
            action(n - 1, m - 1, n - 2, m - 1, n - 1, m - 2);
    }

    static void removeRight(int x) {
        int y = m - 1;
        if (map[x][y] + map[x + 1][y] == 2)
            action(x, y, x + 1, y, x, y - 1);
        else if (map[x][y] == 1)
            action(x, y, x, y - 1, x + 1, y - 1);
        else if (map[x + 1][y] == 1)
            action(x + 1, y, x, y - 1, x + 1, y - 1);
    }

    static void removeDown(int y) {
        int x = n - 1;
        if (map[x][y] + map[x][y + 1] == 2)
            action(x, y, x, y + 1, x - 1, y);
        else if (map[x][y] == 1)
            action(x, y, x - 1, y, x - 1, y + 1);
        else if (map[x][y + 1] == 1)
            action(x, y + 1, x - 1, y, x - 1, y + 1);
    }

    static void action(int x1, int y1, int x2, int y2, int x3, int y3) {
        actions.add((x1 + 1) + " " + (y1 + 1) + " " + (x2 + 1) + " " + (y2 + 1) + " " + (x3 + 1) + " " + (y3 + 1));
        invert(x1, y1);
        invert(x2, y2);
        invert(x3, y3);
    }
}



