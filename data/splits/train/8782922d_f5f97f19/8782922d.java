import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;

public class Interesting_Story {
    public static void main(String[] args) {
        FastScanner sc = new FastScanner();
        int t = sc.nextInt();
        while (t-- > 0) {
            int n = sc.nextInt();
            List<String> l = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                l.add(sc.next());
            }
            int res = 0;
            for (int i = 0; i <= 5; i++) {
                List<Integer> myL = new ArrayList<>();
                for (String s : l) {
                    int count1 = 0, count2 = 0;
                    for (char ch : s.toCharArray()) {
                        if (ch == (char) (i + 97)) {
                            count1++;
                        } else {
                            count2++;
                        }
                    }
                    myL.add(count2 - count1);
                }
                //System.out.println(myL);
                Collections.sort(myL);
                int sum = 0;
                for (int m = 0; m < myL.size(); m++) {
                    sum += myL.get(m);
                    if (sum < 0) {
                        res = Math.max(res, m + 1);
                    }
                }
            }
            System.out.println(res);
        }
    }
}
