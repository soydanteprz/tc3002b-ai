import java.util.*;

public class A {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        StringBuilder sb = new StringBuilder();
        int testCases = sc.nextInt();

        for (int t = 0; t < testCases; t++) {
            int n = sc.nextInt();
            int m = sc.nextInt();

            char c[][] = new char[n][m];

            for (int i = 0; i < n; i++) {
                String str = sc.next();
                c[i] = str.toCharArray();
            }

            int count = 0;
            StringBuilder ans = new StringBuilder();

            for (int i = 0; i < m - 2; i++) {
                for (int j = 0; j < n; j++) {

                    if (c[j][i] == '0') {
                        continue;
                    }
                    count++;

                    if (j == n - 1) {

                        c[j][i] = '0';
                        c[j][i + 1] = (c[j][i + 1] == '0' ? '1' : '0');
                        c[j - 1][i + 1] = (c[j - 1][i + 1] == '0' ? '1' : '0');

                        ans.append((j + 1) + " " + (i + 1) + " " + (j + 1) + " " + (i + 2) + " " + (j) + " " + (i + 2))
                                .append("\n");
                        continue;
                    }

                    c[j][i] = '0';
                    c[j][i + 1] = (c[j][i + 1] == '0' ? '1' : '0');
                    c[j + 1][i + 1] = (c[j + 1][i + 1] == '0' ? '1' : '0');

                    ans.append((j + 1) + " " + (i + 1) + " " + (j + 1) + " " + (i + 2) + " " + (j + 2) + " " + (i + 2))
                            .append("\n");

                }
            }

            for (int i = 0; i < n - 2; i++) {
                for (int j = m - 2; j < m; j++) {
                    if (c[i][j] == '0') {
                        continue;
                    }
                    count++;

                    if (j == m - 1) {
                        c[i][j] = '0';
                        c[i + 1][j] = (c[i + 1][j] == '0' ? '1' : '0');
                        c[i + 1][j - 1] = (c[i + 1][j - 1] == '0' ? '1' : '0');

                        ans.append((i + 1) + " " + (j + 1) + " " + (i + 2) + " " + (j + 1) + " " + (i + 2) + " " + (j))
                                .append("\n");

                        continue;
                    }

                    c[i][j] = '0';
                    c[i + 1][j] = (c[i + 1][j] == '0' ? '1' : '0');
                    c[i + 1][j + 1] = (c[i + 1][j + 1] == '0' ? '1' : '0');

                    ans.append((i + 1) + " " + (j + 1) + " " + (i + 2) + " " + (j + 1) + " " + (i + 2) + " " + (j + 2))
                            .append("\n");
                }
            }

            ArrayList<String> al = new ArrayList<>();

            int count_one = 0;

            for (int i = n - 2; i < n; i++) {
                for (int j = m - 2; j < m; j++) {

                    if (c[i][j] == '0') {
                        al.add((i + 1) + " " + (j + 1));
                    } else {
                        count_one++;
                        al.add(0, (i + 1) + " " + (j + 1));
                    }

                }
            }

            if (count_one == 0) {
                sb.append(count).append("\n");
                sb.append(ans).append("\n");
                continue;
            }

            if (count_one == 1) {
                ans.append(al.get(0) + " " + al.get(1) + " " + al.get(2)).append("\n");
                ans.append(al.get(0) + " " + al.get(2) + " " + al.get(3)).append("\n");
                ans.append(al.get(0) + " " + al.get(1) + " " + al.get(3)).append("\n");

                count = count + 3;

                sb.append(count).append("\n");
                sb.append(ans).append("\n");
                continue;
            }

            if (count_one == 2) {
                ans.append(al.get(0) + " " + al.get(2) + " " + al.get(3)).append("\n");
                ans.append(al.get(1) + " " + al.get(2) + " " + al.get(3)).append("\n");

                count = count + 2;
                sb.append(count).append("\n");
                sb.append(ans).append("\n");
                continue;
            }

            if (count_one == 3) {
                ans.append(al.get(0) + " " + al.get(1) + " " + al.get(2)).append("\n");

                count = count + 1;
                sb.append(count).append("\n");
                sb.append(ans).append("\n");
                continue;
            }

            if (count_one == 4) {
                ans.append(al.get(1) + " " + al.get(2) + " " + al.get(3)).append("\n");

                ans.append(al.get(0) + " " + al.get(1) + " " + al.get(2)).append("\n");
                ans.append(al.get(0) + " " + al.get(2) + " " + al.get(3)).append("\n");
                ans.append(al.get(0) + " " + al.get(1) + " " + al.get(3)).append("\n");

                count = count + 4;

                sb.append(count).append("\n");
                sb.append(ans).append("\n");
                continue;

            }

        }

        System.out.print(sb);
    }
}
