import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import java.util.StringJoiner;

public class D {

  private static List<Long> chairs;
  private static List<Long> folks;
  private static Long[] data;
  private static Long[][] cache;

  public static void main(String[] args) {
    Print print = new Print();
    Scan scan = new Scan();

    int n = scan.scanInt();
    data = scan.scan1dLongArray();
    chairs = new ArrayList<>();
    folks = new ArrayList<>();
    cache = new Long[n][n];

    for (int i = 0; i < n; i++) {
      if (data[i] == 0) {
        chairs.add((long) i);
      } else {
        folks.add((long) i);
      }
    }

    print.printLine(Long.toString(solve(folks, chairs, 0, 0)));
    print.close();
  }

  private static long solve(List<Long> folks, List<Long> chairs, int i, int j) {
    if (i == folks.size()) {
      return 0;
    }
    if (j == chairs.size()) {
      return Integer.MAX_VALUE;
    }

    if (cache[i][j] != null) {
      return cache[i][j];
    }

    return cache[i][j] = Math
        .min(Math.abs(folks.get(i) - chairs.get(j)) + solve(folks, chairs, i + 1, j + 1),
            solve(folks, chairs, i, j + 1));

  }


  static class Scan {

    private byte[] buf = new byte[1024];
    private int index;
    private InputStream in;
    private int total;

    public Scan() {
      in = System.in;
    }

    public int scan() {
      if (total < 0) {
        throw new InputMismatchException();
      }
      if (index >= total) {
        index = 0;
        try {
          total = in.read(buf);
        } catch (IOException ignored) {
        }
        if (total <= 0) {
          return -1;
        }
      }
      return buf[index++];
    }

    public int scanInt() {
      int integer = 0;
      int n = scan();
      while (isWhiteSpace(n)) {
        n = scan();
      }
      int neg = 1;
      if (n == '-') {
        neg = -1;
        n = scan();
      }
      while (!isWhiteSpace(n)) {
        if (n >= '0' && n <= '9') {
          integer *= 10;
          integer += n - '0';
          n = scan();
        } else {
          throw new InputMismatchException();
        }
      }
      return neg * integer;
    }

    

    

    

    

    

    

    

    public Long[] scan1dLongArray() {
      String[] s = this.scanString().split(" ");
      Long[] arr = new Long[s.length];
      for (int i = 0; i < s.length; i++) {
        arr[i] = Long.parseLong(s[i]);
      }
      return arr;
    }

    

    public String scanString() {
      StringBuilder sb = new StringBuilder();
      int n = scan();
      while (isWhiteSpace(n)) {
        n = scan();
      }
      while (!isWhiteSpace(n)) {
        sb.append((char) n);
        n = scan();
      }
      return sb.toString();
    }

    private boolean isWhiteSpace(int n) {
      if (n == '\n' || n == '\r' || n == '\t' || n == -1) {
        return true;
      }
      return false;
    }
  }

  

}
