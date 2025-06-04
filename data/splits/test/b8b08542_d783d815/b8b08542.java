import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.PriorityQueue;
import java.util.Scanner;
import java.util.StringJoiner;
import java.util.StringTokenizer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

import static java.lang.System.out;
import static java.util.stream.Collectors.joining;


public class C {
  static FastScanner sc = new FastScanner(System.in);

  public static void main(String[] args) {
    try (PrintWriter out = new PrintWriter(System.out)) {
      int T = sc.nextInt();
      for (int tt = 1; tt <= T; tt++) {
        int n = sc.nextInt();
        String[] strings = new String[n];
        PriorityQueue<Pair>[] frequencies = new PriorityQueue[5];
        for (int i = 0; i < 5; i++)
          frequencies[i] = new PriorityQueue<>();
        for (int i = 0; i < n; i++) {
          strings[i] = sc.next();
          int[] freq = new int[5];
          for (char c : strings[i].toCharArray())
            freq[c - 'a']++;
          for (int j = 0; j < 5; j++) {
            frequencies[j].add(new Pair(freq[j], strings[i].length()));
          }
        }
        int ans = 0;
        for (PriorityQueue<Pair> pq : frequencies) {
          long curlen = 0;
          long curfreq = 0;
          int cnt = 0;
          while (!pq.isEmpty()) {
            Pair pair = pq.remove();
            curfreq += pair.freq;
            curlen += pair.len;
            if (curfreq >= curlen / 2 + 1) {
              cnt++;
            } else {
              break;
            }
          }
          ans = Math.max(ans, cnt);
        }
        System.out.println(ans);
      }
    }

  }

  static class Pair implements Comparable<Pair> {
    int freq, len;

    public Pair(int freq, int len) {
      this.freq = freq;
      this.len = len;
    }

    

    
  }
  /*
1  
5
cbdca
d
a
d
eb
   */

  
}
