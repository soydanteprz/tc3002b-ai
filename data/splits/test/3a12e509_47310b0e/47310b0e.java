import java.io.*;
import java.util.*;

public class Codeforces {
    public static void main(String[] args) throws IOException{
        try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
            int t = Integer.parseInt(br.readLine());
            while (t-- > 0) {
                int n = Integer.parseInt(br.readLine());
                String[] s = br.readLine().split(" ");
                String c = br.readLine();

                ArrayList<Integer> blue = new ArrayList<>();
                ArrayList<Integer> red = new ArrayList<>();
                
                for (int i = 0; i < n; i++) {
                    if (c.charAt(i) == 'B') blue.add(Integer.parseInt(s[i]));
                    else
                        red.add(Integer.parseInt(s[i]));
                }

                Collections.sort(blue);
                Collections.sort(red);
                int p1 = 0, p2 = 0, num = 1, flag = 1;
                while (num <= n) {
                    if (p1 < blue.size()) {
                        if (blue.get(p1) < num) {
                            flag = 0;
                            break;
                        }
                        p1++;
                    }
                    else if (p2 < red.size()) {
                        if (red.get(p2) > num) {
                            flag = 0;
                            break;
                        }
                        p2++;
                    }
                    num++;
                }
                if (flag == 1)  System.out.println("YES");
                else
                    System.out.println("NO");
            }
        }
    }
}
