import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        int t = nextInt();
        while (t-- != 0) {
            int n = nextInt();
            int[] a = new int[n];
            for (int i = 0; i < n; i++) {
                a[i] = nextInt();
            }
            String s = nextToken();
            ArrayList<Integer> r = new ArrayList<>();
            ArrayList<Integer> b = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (s.charAt(i) == 'B') b.add(a[i]);
                else r.add(a[i]);
            }
            Collections.sort(b);
            Collections.sort(r);
            int y = 1;
            boolean e = true;
            for (int i = 0; i < b.size(); i++) {
                if (b.get(i) < y) {
                    e = false;
                    break;
                }
                y++;
            }
            for (int i = 0; i < r.size(); i++) {
                if (r.get(i) > y) {
                    e = false;
                    break;
                }
                y++;
            }
            if(e)out.println("YES");
            else out.println("NO");
        }
        out.close();
    }

    static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    static PrintWriter out = new PrintWriter(System.out);
    static StringTokenizer in = new StringTokenizer("");


    

    public static String nextToken() throws IOException {
        while (!in.hasMoreTokens()) {
            in = new StringTokenizer(br.readLine());
        }
        return in.nextToken();
    }

    public static int nextInt() throws IOException {
        return Integer.parseInt(nextToken());
    }

    

    
}
