import java.util.*;
import java.io.*;

public class codeforces {
    

    static FastReader fr;
    static StringBuilder res;

    

    public static void main(String[] args) throws java.lang.Exception {
        fr = new FastReader();
        boolean fixed = false;
        int t = !fixed ? fr.nextInt() : 1;
        res = new StringBuilder();
        while (t-- > 0) {
            solve();
        }
        fr.write(res.toString());
        fr.close();
    }

    static int M = 1000008;
    static boolean[] primes = new boolean[M];
    static int[] dp = new int[M];

    

    public static void solve() {
        int n = fr.nextInt();
        String[] arr = new String[n];
        for (int i = 0; i < n; i++) {
            arr[i] = fr.nextLine();
        }
        char[] chars = { 'a', 'b', 'c', 'd', 'e' };
        int ans = 0;
        for (int i = 0; i < 5; i++) {
            char ch = chars[i];
            List<Integer> a = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                String s = arr[j];
                int c = 0;
                for (int k = 0; k < s.length(); k++) {
                    if (s.charAt(k) == ch)
                        c++;
                }
                a.add(2 * c - s.length());
            }
            // System.out.println(a);
            Collections.sort(a, Collections.reverseOrder());
            int sum = 0, len = 0;
            for (int k = 0; k < a.size(); k++) {
                if (sum + a.get(k) <= 0) {
                    break;
                } else {
                    len++;
                    sum += a.get(k);
                }
            }
            ans = Math.max(ans, len);
        }
        res.append(ans + "\n");
    }

    

    

    
}
