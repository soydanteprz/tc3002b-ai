import java.io.*;
import java.util.*;

public class MainClass {
    public static void main(String[] args) {
        Reader in = new Reader(System.in);
        int t = in.nextInt();
        StringBuilder stringBuilder = new StringBuilder();
        while (t-- > 0) {
            ArrayList<Integer> reds = new ArrayList<>();
            ArrayList<Integer> blue = new ArrayList<>();
            int n = in.nextInt();
            int[] a = new int[n];
            for (int i = 0; i < n; i++) {
                a[i] = in.nextInt() - 1;
            }
            char[] s = in.next().toCharArray();
            for (int i = 0; i < n; i++) {
                if (s[i] == 'R') {
                    reds.add(a[i]);
                } else {
                    blue.add(a[i]);
                }
            }
            Collections.sort(reds, Collections.reverseOrder());
            Collections.sort(blue);
            boolean ff = true;
            int start = 0;
            for (int i = 0; i < blue.size(); i++) {
                if (blue.get(i) < start) {
                    ff = false;
                    break;
                }
                start++;
            }
            start = n - 1;
            for (int i = 0; i < reds.size(); i++) {
                if (reds.get(i) > start) {
                    ff = false;
                    break;
                }
                start--;
            }
            stringBuilder.append(ff?"YES":"NO").append("\n");
        }
        System.out.println(stringBuilder);
    }
}

