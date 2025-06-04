import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;

public class C2 {
    static ArrayList<Integer[]> list;

    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        int t = in.nextInt();
        for (int i = 0; i < t; i++) {
            list = new ArrayList<>();
            int n = in.nextInt();
            int m = in.nextInt();
            char[][] arr = new char[n][m];
            for (int j = 0; j < n; j++) {
                arr[j] = in.next().toCharArray();
            }
            for (int j = 0; j < n; j += 2) {
                for (int k = 0; k < m; k += 2) {
                    if (j == n - 1) j = n - 2;
                    if (k == m - 1) k = m - 2;
                    int cnt = 0;
                    while (arr[j][k] == '1' || arr[j + 1][k] == '1' || arr[j][k + 1] == '1' || arr[j + 1][k + 1] == '1') {
//                        System.out.println(arr[j][k] + " " + arr[j][k + 1] + "\n" + arr[j + 1][k] + " " + arr[j + 1][k + 1]);

                        if (checkThree(arr, j, k)) {
//                            System.out.println("3:");
//                            System.out.println(arr[j][k] + " " + arr[j][k + 1] + "\n" + arr[j + 1][k] + " " + arr[j + 1][k + 1]);
                            continue;
                        } else if (checkTwo(arr, j, k)) {
//                            System.out.println("2:");
//                            System.out.println(arr[j][k] + " " + arr[j][k + 1] + "\n" + arr[j + 1][k] + " " + arr[j + 1][k + 1]);
                            checkThree(arr, j, k);
//                            System.out.println("3:");
//                            System.out.println(arr[j][k] + " " + arr[j][k + 1] + "\n" + arr[j + 1][k] + " " + arr[j + 1][k + 1]);
                            continue;
                        } else if (checkDiagonal(arr, j, k)) {
//                            System.out.println("/2:");
//                            System.out.println(arr[j][k] + " " + arr[j][k + 1] + "\n" + arr[j + 1][k] + " " + arr[j + 1][k + 1]);
                        } else if (checkOne(arr, j, k)) {
//                            System.out.println("1:");
//                            System.out.println(arr[j][k] + " " + arr[j][k + 1] + "\n" + arr[j + 1][k] + " " + arr[j + 1][k + 1]);
                        }
//                        cnt++;
//                        if (cnt == 10) return;
//                        System.out.println();
                    }
//                    System.out.println("qwe");
//                    System.out.println("qwe");
                }
            }
            out.println(list.size());
            for (Integer[] integers : list) {
                for (int j = 0; j < integers.length; j++) {
                    out.print((integers[j] + 1) + " ");
                }
                out.println();
            }
        }
        out.close();
    }

    public static boolean checkThree(char[][] arr, int i, int j) {
        if (arr[i][j] == '1' && arr[i][j + 1] == '1' && arr[i + 1][j] == '1') {
            arr[i][j] = '0';
            arr[i][j + 1] = '0';
            arr[i + 1][j] = '0';
            list.add(new Integer[]{i, j, i, (j + 1), (i + 1), j});
            return true;
        } else if (arr[i][j] == '1' && arr[i][j + 1] == '1' && arr[i + 1][j + 1] == '1') {
            arr[i][j] = '0';
            arr[i][j + 1] = '0';
            arr[i + 1][j + 1] = '0';
            list.add(new Integer[]{i, j, i, (j + 1), (i + 1), j + 1});
            return true;
        } else if (arr[i + 1][j] == '1' && arr[i + 1][j + 1] == '1' && arr[i][j] == '1') {
            arr[i + 1][j] = '0';
            arr[i + 1][j + 1] = '0';
            arr[i][j] = '0';
            list.add(new Integer[]{i + 1, j, i + 1, j + 1, (i), j});
            return true;
        } else if (arr[i + 1][j] == '1' && arr[i + 1][j + 1] == '1' && arr[i][j + 1] == '1') {
            arr[i + 1][j] = '0';
            arr[i + 1][j + 1] = '0';
            arr[i][j + 1] = '0';
            list.add(new Integer[]{i + 1, j, i + 1, j + 1, (i), j + 1});
            return true;
        }
        return false;
    }

    public static boolean checkDiagonal(char[][] arr, int i, int j) {
        if (arr[i][j] == '1' && arr[i + 1][j + 1] == '1') {
            arr[i][j] = '0';
            arr[i][j + 1] = '1';
            arr[i + 1][j] = '1';
            list.add(new Integer[]{i, j, i, j + 1, (i + 1), j});
            return true;
        } else if (arr[i + 1][j] == '1' && arr[i][j + 1] == '1') {
            arr[i + 1][j] = '0';
            arr[i][j] = '1';
            arr[i + 1][j + 1] = '1';
            list.add(new Integer[]{i + 1, j, i, j, (i + 1), j + 1});
            return true;
        }
        return false;
    }

    public static boolean checkOne(char[][] arr, int i, int j) {
        if (arr[i][j] == '1') {
            arr[i][j] = '0';
            arr[i + 1][j] = '1';
            arr[i][j + 1] = '1';
            list.add(new Integer[]{i, j, i + 1, j, (i), j + 1});
            return true;
        } else if (arr[i + 1][j] == '1') {
            arr[i + 1][j] = '0';
            arr[i][j] = '1';
            arr[i + 1][j + 1] = '1';
            list.add(new Integer[]{i + 1, j, i, j, (i + 1), j + 1});
            return true;
        } else if (arr[i][j + 1] == '1') {
            arr[i][j + 1] = '0';
            arr[i][j] = '1';
            arr[i + 1][j + 1] = '1';
            list.add(new Integer[]{i, j + 1, i, j, (i + 1), j + 1});
            return true;
        } else if (arr[i + 1][j + 1] == '1') {
            arr[i + 1][j + 1] = '0';
            arr[i + 1][j] = '1';
            arr[i][j + 1] = '1';
            list.add(new Integer[]{i + 1, j + 1, i + 1, j, (i), j + 1});
            return true;
        }
        return false;
    }

    public static char changeValue(char c) {
        return c == '1' ? '0' : '1';
    }

    public static boolean checkTwo(char[][] arr, int i, int j) {
        if (arr[i][j] == '1' && arr[i][j + 1] == '1') {
            arr[i][j] = '0';
            arr[i + 1][j + 1] = changeValue(arr[i + 1][j + 1]);
            arr[i + 1][j] = changeValue(arr[i + 1][j]);
            list.add(new Integer[]{i, j, i + 1, j + 1, (i + 1), j});
            return true;
        } else if (arr[i][j] == '1' && arr[i + 1][j] == '1') {
            arr[i][j] = '0';
            arr[i + 1][j + 1] = changeValue(arr[i + 1][j + 1]);
            arr[i][j + 1] = changeValue(arr[i][j + 1]);
            list.add(new Integer[]{i, j, i + 1, j + 1, (i), j + 1});
            return true;
        } else if (arr[i + 1][j] == '1' && arr[i + 1][j + 1] == '1') {
            arr[i + 1][j] = '0';
            arr[i][j + 1] = changeValue(arr[i][j + 1]);
            arr[i][j] = changeValue(arr[i][j]);
            list.add(new Integer[]{i + 1, j, i, j + 1, (i), j});
            return true;
        } else if (arr[i][j + 1] == '1' && arr[i + 1][j + 1] == '1') {
            arr[i][j + 1] = '0';
            arr[i][j] = changeValue(arr[i][j]);
            arr[i + 1][j] = changeValue(arr[i + 1][j]);
            list.add(new Integer[]{i, j + 1, i, j, (i + 1), j});
            return true;
        }
        return false;
    }

    
}
