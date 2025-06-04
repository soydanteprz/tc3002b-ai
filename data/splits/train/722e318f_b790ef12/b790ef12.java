import java.io.*;
import java.util.*;
public class Main {
    

    public static void main(String[] args) {
        FastReader obj = new FastReader();
        PrintWriter out = new PrintWriter(System.out);
        int l = obj.nextInt();
        while (l-- != 0) {
            int n = obj.nextInt();
            int[] num = new int[n];
            for (int i = 0; i < n; i++) num[i] = obj.nextInt();
            Vector<Integer> red = new Vector<>();
            Vector<Integer> blue = new Vector<>();
            String s = obj.next();
            for (int i = 0; i < s.length(); i++) {
                if (s.charAt(i) == 'R') red.add(num[i]);
                else blue.add(num[i]);
            }
            Collections.sort(blue);
            Collections.sort(red);
            int c = 1, f = 0;
            for (int i = 0; i < blue.size(); i++) {
                if (blue.get(i) < c) {
                    f = 1;
                    break;
                }
                c++;
            }
            for (int i = 0; i < red.size(); i++) {
                if (red.get(i) > c) {
                    f = 1;
                    break;
                }
                c++;
            }
            if (f == 0) out.println("YES");
            else out.println("NO");
        }
        out.flush();
    }
}
