import java.util.*;
import java.lang.*;
import java.io.*;

public class Solution {
    public static void main(String[] args) throws java.lang.Exception {
        out = new PrintWriter(new BufferedOutputStream(System.out));
        sc = new FastReader();

        int test = sc.nextInt();
        for (int t = 0; t < test; t++) {
            solve();
        }
        out.close();
    }

    private static void solve() {
        int n = sc.nextInt();
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = sc.nextInt();
        }

        int[] res = new int[n];
        if (n % 2 == 0) {
            for (int i = 1; i < n; i += 2) {
                res[i] = arr[i - 1];
                res[i - 1] = -arr[i];
            }
        }else {
            for (int i = 4; i < n; i += 2) {
                res[i] = arr[i - 1];
                res[i - 1] = -arr[i];
            }

            if (arr[0] + arr[1] != 0) {
                res[0] = -arr[2];
                res[1] = -arr[2];
                res[2] = arr[0] + arr[1];
            }else if (arr[0] + arr[2] != 0) {
                res[0] = -arr[1];
                res[2] = -arr[1];
                res[1] = arr[0] + arr[2];
            }else {
                res[1] = -arr[0];
                res[2] = -arr[0];
                res[0] = arr[1] + arr[2];
            }
        }

        for (int i = 0; i < n; i++) {
            out.print(res[i] + " ");
        }
        out.println();
    }


    public static FastReader sc;
    public static PrintWriter out;
    
}
